"""Intent state machine — maps gaze + gesture to window actions.

Uses time.monotonic() exclusively for all timing.
Pinch-release debounce: N consecutive non-PINCH frames required before
registering a pinch-end (prevents false double-pinch on classifier blips).
"""
from __future__ import annotations

import time

from gazecontrol.settings import get_settings

GESTURE_ACTION_MAP: dict[str, str] = {
    "GRAB": "DRAG",
    "RELEASE": "RELEASE",
    "PINCH": "RESIZE",
    "SWIPE_LEFT": "MINIMIZE",
    "SWIPE_RIGHT": "BRING_FRONT",
    "CLOSE_SIGN": "CLOSE",
    "SCROLL_UP": "SCROLL_UP",
    "SCROLL_DOWN": "SCROLL_DOWN",
    "MAXIMIZE": "MAXIMIZE",
}

CONTINUOUS_GESTURES: set[str] = {"GRAB", "PINCH"}


class IntentStateMachine:
    """Finite state machine: IDLE → TARGETING → READY → ACTING → COOLDOWN."""

    def __init__(self) -> None:
        s = get_settings()
        self._cfg = s.intent
        self._gesture_cfg = s.gesture

        self.state: str = "IDLE"
        self._target_window: dict | None = None
        self._dwell_start: float | None = None
        self._ready_start: float | None = None
        self._cooldown_start: float | None = None
        self._drag_start_hand: tuple[float, float] | None = None
        self._drag_start_rect: tuple | None = None
        self._resize_start_hand: tuple[float, float] | None = None
        self._resize_start_rect: tuple | None = None
        self._active_gesture: str | None = None

        # Double-pinch detection.
        self._pinch_active: bool = False
        self._last_pinch_end_time: float | None = None
        # Debounce: consecutive non-PINCH frames required to confirm pinch release.
        self._non_pinch_frames: int = 0

    def update(
        self,
        gaze_point: tuple[int, int] | None,
        target_window: dict | None,
        gesture_id: str | None,
        gesture_confidence: float,
        hand_position: tuple[float, float] | None = None,
    ) -> dict | None:
        """Run one FSM tick.  Returns an action dict or None."""
        now = time.monotonic()
        cfg = self._cfg
        conf_thr = self._gesture_cfg.confidence_threshold
        release_frames = self._gesture_cfg.pinch_release_frames

        # ------------------------------------------------------------------
        # Double-pinch detection — fires in any FSM state.
        # Requires N≥pinch_release_frames consecutive non-PINCH frames before
        # a new pinch is counted as a second pinch (prevents classifier blips).
        # ------------------------------------------------------------------
        is_pinch = gesture_id == "PINCH" and gesture_confidence > conf_thr
        if is_pinch:
            self._non_pinch_frames = 0
            if not self._pinch_active:
                self._pinch_active = True
                if (
                    self._last_pinch_end_time is not None
                    and now - self._last_pinch_end_time < cfg.double_pinch_window_s
                ):
                    self._pinch_active = False
                    self._last_pinch_end_time = None
                    return {"type": "CLOSE_APP"}
        elif self._pinch_active:
            self._non_pinch_frames += 1
            if self._non_pinch_frames >= release_frames:
                self._pinch_active = False
                self._last_pinch_end_time = now
                self._non_pinch_frames = 0

        # ------------------------------------------------------------------
        # Main FSM
        # ------------------------------------------------------------------
        if self.state == "IDLE":
            if target_window and gaze_point:
                self._target_window = target_window
                self._dwell_start = now
                self.state = "TARGETING"
            return None

        if self.state == "TARGETING":
            if not target_window or (
                self._target_window
                and target_window.get("hwnd") != self._target_window.get("hwnd")
            ):
                self.state = "IDLE"
                self._target_window = None
                self._dwell_start = None
                return None
            elapsed = now - (self._dwell_start or now)
            if elapsed >= cfg.dwell_time_s:
                self.state = "READY"
                self._ready_start = now
            return None

        if self.state == "READY":
            if now - (self._ready_start or now) > cfg.ready_timeout_s:
                self.state = "IDLE"
                self._target_window = None
                self._ready_start = None
                return None
            if gesture_id and gesture_confidence > conf_thr:
                action_type = GESTURE_ACTION_MAP.get(gesture_id)
                if not action_type or action_type == "RELEASE":
                    return None
                self.state = "ACTING"
                self._active_gesture = gesture_id
                if gesture_id == "GRAB":
                    self._drag_start_hand = hand_position
                    self._drag_start_rect = self._target_window.get("rect") if self._target_window else None
                    return {"type": "DRAG", "window": self._target_window, "data": {"phase": "start"}}
                if gesture_id == "PINCH":
                    self._resize_start_rect = self._target_window.get("rect") if self._target_window else None
                    self._resize_start_hand = hand_position
                    return {"type": "RESIZE", "window": self._target_window, "data": {"phase": "start"}}
                action = {"type": action_type, "window": self._target_window, "data": {}}
                self.state = "COOLDOWN"
                self._cooldown_start = now
                self._active_gesture = None
                return action
            return None

        if self.state == "ACTING":
            if self._active_gesture == "GRAB":
                if gesture_id == "RELEASE" or gesture_id != "GRAB":
                    self.state = "COOLDOWN"
                    self._cooldown_start = now
                    self._active_gesture = None
                    self._drag_start_hand = None
                    self._drag_start_rect = None
                    return {"type": "DRAG", "window": self._target_window, "data": {"phase": "end"}}
                if hand_position and self._drag_start_hand:
                    dx = hand_position[0] - self._drag_start_hand[0]
                    dy = hand_position[1] - self._drag_start_hand[1]
                    return {
                        "type": "DRAG",
                        "window": self._target_window,
                        "data": {"phase": "move", "delta": (dx, dy), "start_rect": self._drag_start_rect},
                    }
                return None

            if self._active_gesture == "PINCH":
                if gesture_id != "PINCH":
                    self.state = "COOLDOWN"
                    self._cooldown_start = now
                    self._active_gesture = None
                    self._resize_start_rect = None
                    self._resize_start_hand = None
                    return {"type": "RESIZE", "window": self._target_window, "data": {"phase": "end"}}
                if hand_position and self._resize_start_hand:
                    dx = hand_position[0] - self._resize_start_hand[0]
                    dy = hand_position[1] - self._resize_start_hand[1]
                    return {
                        "type": "RESIZE",
                        "window": self._target_window,
                        "data": {"phase": "move", "delta": (dx, dy), "start_rect": self._resize_start_rect},
                    }
                return None

            self.state = "COOLDOWN"
            self._cooldown_start = now
            self._active_gesture = None
            return None

        if self.state == "COOLDOWN":
            if now - (self._cooldown_start or now) >= cfg.cooldown_s:
                self.state = "IDLE"
                self._target_window = None
                self._cooldown_start = None
            return None

        return None
