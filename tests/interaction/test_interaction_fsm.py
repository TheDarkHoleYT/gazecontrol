"""Tests for InteractionFSM — timeline-driven with a fake clock."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gazecontrol.gesture.pinch_tracker import PinchEvent
from gazecontrol.interaction.interaction_fsm import InteractionFSM
from gazecontrol.interaction.types import HoveredWindow, InteractionKind


def _make_fsm(**kwargs) -> InteractionFSM:
    defaults = dict(tap_ms=200, hold_ms=300, double_ms=400, tap_max_move_px=20, grip_ratio=0.18)
    defaults.update(kwargs)
    return InteractionFSM(**defaults)


def _make_hitter(window=None) -> MagicMock:
    h = MagicMock()
    h.at.return_value = window
    return h


_WIN = HoveredWindow(hwnd=1, rect=(0, 0, 800, 600), title="Test")
_GRIP_WIN = HoveredWindow(hwnd=2, rect=(0, 0, 100, 100), title="Grip")


# ---------------------------------------------------------------------------
# IDLE → quick tap → CLICK
# ---------------------------------------------------------------------------


def test_tap_produces_click():
    fsm = _make_fsm()
    hitter = _make_hitter()

    with patch("time.monotonic", return_value=0.0):
        r = fsm.update(PinchEvent.DOWN, (100, 100), hitter)
    assert r is None
    assert fsm.state == "PINCH_PENDING"

    with patch("time.monotonic", return_value=0.1):  # 100ms < tap_ms=200ms
        r = fsm.update(PinchEvent.UP, (102, 100), hitter)  # tiny movement ✓
    assert r is not None
    assert r.kind == InteractionKind.CLICK
    assert fsm.state == "IDLE"


# ---------------------------------------------------------------------------
# Double-pinch → TOGGLE_LAUNCHER
# ---------------------------------------------------------------------------


def test_double_tap_produces_toggle_launcher():
    fsm = _make_fsm()
    hitter = _make_hitter()

    # First tap.
    with patch("time.monotonic", return_value=0.0):
        fsm.update(PinchEvent.DOWN, (100, 100), hitter)
    with patch("time.monotonic", return_value=0.1):
        r1 = fsm.update(PinchEvent.UP, (100, 100), hitter)
    assert r1.kind == InteractionKind.CLICK

    # Second tap within double_ms.
    with patch("time.monotonic", return_value=0.2):
        fsm.update(PinchEvent.DOWN, (100, 100), hitter)
    with patch("time.monotonic", return_value=0.3):  # 300ms - 100ms = 200ms < double_ms=400ms
        r2 = fsm.update(PinchEvent.UP, (100, 100), hitter)
    assert r2 is not None
    assert r2.kind == InteractionKind.TOGGLE_LAUNCHER


# ---------------------------------------------------------------------------
# Pinch-hold → DRAG
# ---------------------------------------------------------------------------


def test_hold_on_window_starts_drag():
    fsm = _make_fsm()
    hitter = _make_hitter(window=_WIN)

    with patch("time.monotonic", return_value=0.0):
        fsm.update(PinchEvent.DOWN, (100, 100), hitter)

    # After hold_ms, a HOLD event should trigger DRAG_START.
    with patch("time.monotonic", return_value=0.35):  # 350ms > hold_ms=300ms
        r = fsm.update(PinchEvent.HOLD, (100, 100), hitter)

    assert r is not None
    assert r.kind == InteractionKind.DRAG_START
    assert fsm.state == "DRAGGING"


def test_drag_update_reports_delta():
    fsm = _make_fsm()
    hitter = _make_hitter(window=_WIN)

    with patch("time.monotonic", return_value=0.0):
        fsm.update(PinchEvent.DOWN, (100, 100), hitter)
    with patch("time.monotonic", return_value=0.35):
        fsm.update(PinchEvent.HOLD, (100, 100), hitter)  # DRAG_START

    with patch("time.monotonic", return_value=0.36):
        r = fsm.update(PinchEvent.HOLD, (150, 130), hitter)

    assert r is not None
    assert r.kind == InteractionKind.DRAG_UPDATE
    assert r.data["new_x"] == 0 + 50  # base_x=0, delta_x=+50
    assert r.data["new_y"] == 0 + 30  # base_y=0, delta_y=+30


def test_drag_ends_on_pinch_up():
    fsm = _make_fsm()
    hitter = _make_hitter(window=_WIN)

    with patch("time.monotonic", return_value=0.0):
        fsm.update(PinchEvent.DOWN, (100, 100), hitter)
    with patch("time.monotonic", return_value=0.35):
        fsm.update(PinchEvent.HOLD, (100, 100), hitter)

    with patch("time.monotonic", return_value=0.36):
        r = fsm.update(PinchEvent.UP, (100, 100), hitter)

    assert r is not None
    assert r.kind == InteractionKind.DRAG_END
    assert fsm.state == "COOLDOWN"


# ---------------------------------------------------------------------------
# Pinch-hold in resize grip → RESIZE
# ---------------------------------------------------------------------------


def test_hold_in_grip_starts_resize():
    fsm = _make_fsm()
    hitter = _make_hitter(window=_GRIP_WIN)

    # _GRIP_WIN rect=(0,0,100,100); grip starts at x=82,y=82
    grip_point = (90, 90)

    with patch("time.monotonic", return_value=0.0):
        fsm.update(PinchEvent.DOWN, grip_point, hitter)
    with patch("time.monotonic", return_value=0.35):
        r = fsm.update(PinchEvent.HOLD, grip_point, hitter)

    assert r is not None
    assert r.kind == InteractionKind.RESIZE_START
    assert fsm.state == "RESIZING"


def test_resize_update_clamps_minimum_size():
    fsm = _make_fsm(min_w=200, min_h=150)
    hitter = _make_hitter(window=_GRIP_WIN)
    grip_point = (90, 90)

    with patch("time.monotonic", return_value=0.0):
        fsm.update(PinchEvent.DOWN, grip_point, hitter)
    with patch("time.monotonic", return_value=0.35):
        fsm.update(PinchEvent.HOLD, grip_point, hitter)

    # Move to smaller → clamped to min_w=200, min_h=150
    with patch("time.monotonic", return_value=0.36):
        r = fsm.update(PinchEvent.HOLD, (30, 30), hitter)

    assert r is not None
    assert r.kind == InteractionKind.RESIZE_UPDATE
    assert r.data["new_w"] >= 200
    assert r.data["new_h"] >= 150


# ---------------------------------------------------------------------------
# Scroll fires independently
# ---------------------------------------------------------------------------


def test_scroll_up_fires_in_idle():
    fsm = _make_fsm()
    hitter = _make_hitter(window=_WIN)
    with patch("time.monotonic", return_value=0.0):
        r = fsm.update(PinchEvent.NONE, (100, 100), hitter, two_finger_scroll_delta=120)
    assert r is not None
    assert r.kind == InteractionKind.SCROLL_UP


def test_scroll_down_fires_in_idle():
    fsm = _make_fsm()
    hitter = _make_hitter(window=_WIN)
    with patch("time.monotonic", return_value=0.0):
        r = fsm.update(PinchEvent.NONE, (100, 100), hitter, two_finger_scroll_delta=-120)
    assert r is not None
    assert r.kind == InteractionKind.SCROLL_DOWN


# ---------------------------------------------------------------------------
# Cooldown → IDLE
# ---------------------------------------------------------------------------


def test_cooldown_transitions_to_idle():
    fsm = _make_fsm(cooldown_ms=100)
    hitter = _make_hitter(window=_WIN)

    with patch("time.monotonic", return_value=0.0):
        fsm.update(PinchEvent.DOWN, (100, 100), hitter)
    with patch("time.monotonic", return_value=0.35):
        fsm.update(PinchEvent.HOLD, (100, 100), hitter)
    with patch("time.monotonic", return_value=0.36):
        fsm.update(PinchEvent.UP, (100, 100), hitter)  # → COOLDOWN

    assert fsm.state == "COOLDOWN"

    with patch("time.monotonic", return_value=0.47):  # 110ms after drag end
        fsm.update(PinchEvent.NONE, None, hitter)

    assert fsm.state == "IDLE"


# ---------------------------------------------------------------------------
# Movement cancels tap
# ---------------------------------------------------------------------------


def test_large_movement_cancels_tap():
    fsm = _make_fsm(tap_max_move_px=10)
    hitter = _make_hitter()

    with patch("time.monotonic", return_value=0.0):
        fsm.update(PinchEvent.DOWN, (100, 100), hitter)
    with patch("time.monotonic", return_value=0.05):
        r = fsm.update(PinchEvent.UP, (200, 100), hitter)  # moved 100px > 10px

    # Should return None (tap cancelled) and return to IDLE.
    assert r is None
    assert fsm.state == "IDLE"


# ---------------------------------------------------------------------------
# No window → return to IDLE from hold
# ---------------------------------------------------------------------------


def test_hold_no_window_returns_to_idle():
    fsm = _make_fsm()
    hitter = _make_hitter(window=None)

    with patch("time.monotonic", return_value=0.0):
        fsm.update(PinchEvent.DOWN, (100, 100), hitter)
    with patch("time.monotonic", return_value=0.35):
        r = fsm.update(PinchEvent.HOLD, (100, 100), hitter)

    assert r is None
    assert fsm.state == "IDLE"
