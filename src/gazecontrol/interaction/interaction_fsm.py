"""InteractionFSM — pinch-based finite state machine for hand-only desktop control.

State diagram::

    IDLE ──pinch_down──▶ PINCH_PENDING
         ┌── PINCH_PENDING ──────────────────────────────────────────────────┐
         │  elapsed < T_tap AND movement < TAP_MAX_MOVE AND pinch_up:        │
         │      → CLICK (single); feed DoublePinchDetector                  │
         │      → TOGGLE_LAUNCHER if double-pinch detected                  │
         │                                                                   │
         │  elapsed ≥ T_hold AND pinch still down:                          │
         │      hit-test window at fingertip_screen                         │
         │      if no window         → IDLE                                 │
         │      elif resize grip     → RESIZING                             │
         │      else                 → DRAGGING                              │
         └───────────────────────────────────────────────────────────────────┘
    DRAGGING ──pinch_up──▶ DRAG_END ──▶ COOLDOWN ──▶ IDLE
    RESIZING ──pinch_up──▶ RESIZE_END ──▶ COOLDOWN ──▶ IDLE

Two-finger scroll (index + middle extended, others closed, vertical velocity)
fires independently in any state except DRAGGING/RESIZING (to avoid conflicts).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from gazecontrol.gesture.pinch_tracker import PinchEvent
from gazecontrol.interaction.grip_region import is_in_resize_grip
from gazecontrol.interaction.types import HoveredWindow, Interaction, InteractionKind

if TYPE_CHECKING:
    from gazecontrol.interaction.window_hit_tester import WindowHitTester

logger = logging.getLogger(__name__)


class InteractionFSM:
    """Finite state machine that converts per-frame pinch events into interactions.

    Args:
        tap_ms:         Max milliseconds for a quick tap (tap vs hold discriminator).
        hold_ms:        Milliseconds of continuous pinch before drag/resize begins.
        double_ms:      Max milliseconds between two taps for a double-pinch.
        tap_max_move_px: Max pixel movement during a tap before it is ignored.
        grip_ratio:     Fraction of window dimensions that constitutes the resize grip.
        cooldown_ms:    Milliseconds to stay in COOLDOWN before returning to IDLE.
        min_w:          Minimum allowed window width during resize.
        min_h:          Minimum allowed window height during resize.
    """

    def __init__(
        self,
        *,
        tap_ms: int = 220,
        hold_ms: int = 280,
        double_ms: int = 420,
        tap_max_move_px: int = 18,
        grip_ratio: float = 0.18,
        cooldown_ms: int = 120,
        min_w: int = 200,
        min_h: int = 150,
    ) -> None:
        self._tap_ms = tap_ms
        self._hold_ms = hold_ms
        self._double_ms = double_ms
        self._tap_max_move = tap_max_move_px
        self._grip_ratio = grip_ratio
        self._cooldown_s = cooldown_ms / 1000.0
        self._min_w = min_w
        self._min_h = min_h

        # FSM is documented as single-threaded but stages may indirectly
        # re-enter ``update()`` through callbacks; an RLock keeps each tick's
        # state mutation atomic without dead-locking on re-entry.
        self._lock = threading.RLock()

        self.state: str = "IDLE"

        # PINCH_PENDING scratch
        self._pinch_down_time: float | None = None
        self._pinch_down_point: tuple[int, int] | None = None

        # DRAGGING / RESIZING scratch
        self._active_window: HoveredWindow | None = None
        self._anchor_point: tuple[int, int] | None = None
        self._base_rect: tuple[int, int, int, int] | None = None

        # Cooldown
        self._cooldown_start: float | None = None

        # Double-pinch
        self._last_tap_time: float | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        pinch_event: PinchEvent,
        fingertip_screen: tuple[int, int] | None,
        hit_tester: WindowHitTester,
        two_finger_scroll_delta: int = 0,
    ) -> Interaction | None:
        """Run one FSM tick.

        Args:
            pinch_event:            Event from :class:`~gazecontrol.gesture.pinch_tracker.PinchTracker`.
            fingertip_screen:       Current index-fingertip screen coordinates, or ``None``.
            hit_tester:             A :class:`~gazecontrol.interaction.window_hit_tester.WindowHitTester`
                                    instance (duck-typed to avoid circular imports).
            two_finger_scroll_delta: Non-zero when a two-finger scroll is detected
                                    (positive=up, negative=down).

        Returns:
            An :class:`Interaction` or ``None`` if nothing actionable happened.
        """
        with self._lock:
            return self._update_locked(
                pinch_event, fingertip_screen, hit_tester, two_finger_scroll_delta
            )

    def _update_locked(
        self,
        pinch_event: PinchEvent,
        fingertip_screen: tuple[int, int] | None,
        hit_tester: WindowHitTester,
        two_finger_scroll_delta: int,
    ) -> Interaction | None:
        now = time.monotonic()

        # Two-finger scroll fires in IDLE / PINCH_PENDING — not during drag/resize.
        if (
            two_finger_scroll_delta != 0
            and self.state not in ("DRAGGING", "RESIZING")
            and fingertip_screen is not None
        ):
            window = hit_tester.at(fingertip_screen)
            kind = (
                InteractionKind.SCROLL_UP
                if two_finger_scroll_delta > 0
                else InteractionKind.SCROLL_DOWN
            )
            return Interaction(
                kind=kind,
                window=window,
                point=fingertip_screen,
                data={"delta": two_finger_scroll_delta},
            )

        # ── IDLE ──────────────────────────────────────────────────────────────
        if self.state == "IDLE":
            if pinch_event == PinchEvent.DOWN and fingertip_screen is not None:
                self.state = "PINCH_PENDING"
                self._pinch_down_time = now
                self._pinch_down_point = fingertip_screen
            return None

        # ── COOLDOWN ──────────────────────────────────────────────────────────
        if self.state == "COOLDOWN":
            cooldown_start = self._cooldown_start if self._cooldown_start is not None else now
            if now - cooldown_start >= self._cooldown_s:
                self._transition_to_idle()
            return None

        # ── PINCH_PENDING ─────────────────────────────────────────────────────
        if self.state == "PINCH_PENDING":
            return self._handle_pinch_pending(now, pinch_event, fingertip_screen, hit_tester)

        # ── DRAGGING ──────────────────────────────────────────────────────────
        if self.state == "DRAGGING":
            return self._handle_dragging(pinch_event, fingertip_screen)

        # ── RESIZING ──────────────────────────────────────────────────────────
        if self.state == "RESIZING":
            return self._handle_resizing(pinch_event, fingertip_screen)

        return None

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_pinch_pending(
        self,
        now: float,
        pinch_event: PinchEvent,
        fingertip_screen: tuple[int, int] | None,
        hit_tester: object,
    ) -> Interaction | None:
        down_time = self._pinch_down_time if self._pinch_down_time is not None else now
        down_point = (
            self._pinch_down_point
            if self._pinch_down_point is not None
            else (fingertip_screen or (0, 0))
        )
        elapsed_ms = (now - down_time) * 1000.0

        current_point = fingertip_screen or down_point
        movement_px = max(
            abs(current_point[0] - down_point[0]),
            abs(current_point[1] - down_point[1]),
        )

        # Quick release → TAP
        if pinch_event == PinchEvent.UP:
            self._transition_to_idle()
            if elapsed_ms < self._tap_ms and movement_px <= self._tap_max_move:
                return self._emit_tap(now, current_point)
            return None

        # Held long enough → classify as DRAG or RESIZE
        if elapsed_ms >= self._hold_ms:
            if fingertip_screen is None:
                self._transition_to_idle()
                return None
            window: HoveredWindow | None = hit_tester.at(fingertip_screen)  # type: ignore[attr-defined]
            if window is None:
                self._transition_to_idle()
                return None
            if is_in_resize_grip(window.rect, fingertip_screen, self._grip_ratio):
                self.state = "RESIZING"
            else:
                self.state = "DRAGGING"
            self._active_window = window
            self._anchor_point = fingertip_screen
            self._base_rect = window.rect
            kind = (
                InteractionKind.RESIZE_START
                if self.state == "RESIZING"
                else InteractionKind.DRAG_START
            )
            return Interaction(
                kind=kind, window=window, point=fingertip_screen, data={"phase": "start"}
            )

        return None

    def _emit_tap(self, now: float, point: tuple[int, int]) -> Interaction:
        """Check for double-pinch; return TOGGLE_LAUNCHER or CLICK."""
        if (
            self._last_tap_time is not None
            and (now - self._last_tap_time) * 1000.0 < self._double_ms
        ):
            self._last_tap_time = None  # consume the double-tap
            return Interaction(kind=InteractionKind.TOGGLE_LAUNCHER, point=point)
        self._last_tap_time = now
        return Interaction(kind=InteractionKind.CLICK, point=point)

    def _handle_dragging(
        self,
        pinch_event: PinchEvent,
        fingertip_screen: tuple[int, int] | None,
    ) -> Interaction | None:
        if pinch_event == PinchEvent.UP or pinch_event == PinchEvent.NONE:
            window = self._active_window
            self._start_cooldown()
            return Interaction(
                kind=InteractionKind.DRAG_END,
                window=window,
                point=fingertip_screen or (0, 0),
                data={"phase": "end"},
            )

        if (
            fingertip_screen is not None
            and self._anchor_point is not None
            and self._base_rect is not None
        ):
            dx = fingertip_screen[0] - self._anchor_point[0]
            dy = fingertip_screen[1] - self._anchor_point[1]
            base_x, base_y, _, _ = self._base_rect
            return Interaction(
                kind=InteractionKind.DRAG_UPDATE,
                window=self._active_window,
                point=fingertip_screen,
                data={"new_x": base_x + dx, "new_y": base_y + dy, "delta": (dx, dy)},
            )
        return None

    def _handle_resizing(
        self,
        pinch_event: PinchEvent,
        fingertip_screen: tuple[int, int] | None,
    ) -> Interaction | None:
        if pinch_event == PinchEvent.UP or pinch_event == PinchEvent.NONE:
            window = self._active_window
            self._start_cooldown()
            return Interaction(
                kind=InteractionKind.RESIZE_END,
                window=window,
                point=fingertip_screen or (0, 0),
                data={"phase": "end"},
            )

        if (
            fingertip_screen is not None
            and self._anchor_point is not None
            and self._base_rect is not None
        ):
            dx = fingertip_screen[0] - self._anchor_point[0]
            dy = fingertip_screen[1] - self._anchor_point[1]
            base_x, base_y, base_w, base_h = self._base_rect
            new_w = max(self._min_w, base_w + dx)
            new_h = max(self._min_h, base_h + dy)
            return Interaction(
                kind=InteractionKind.RESIZE_UPDATE,
                window=self._active_window,
                point=fingertip_screen,
                data={"new_x": base_x, "new_y": base_y, "new_w": new_w, "new_h": new_h},
            )
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _start_cooldown(self) -> None:
        self.state = "COOLDOWN"
        self._cooldown_start = time.monotonic()
        self._active_window = None
        self._anchor_point = None
        self._base_rect = None

    def _transition_to_idle(self) -> None:
        self.state = "IDLE"
        self._pinch_down_time = None
        self._pinch_down_point = None
        self._active_window = None
        self._anchor_point = None
        self._base_rect = None
        self._cooldown_start = None
