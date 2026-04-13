"""Tests for IntentStateMachine — FSM transition coverage.

Fixed: removed unbound `cfg` references (original bug).
All timing now uses time.monotonic() consistently with the FSM implementation.
"""
from __future__ import annotations

import time

from gazecontrol.intent.state_machine import IntentStateMachine
from gazecontrol.settings import get_settings

WINDOW_A = {"hwnd": 1001, "title": "Test", "rect": (100, 100, 400, 300)}
WINDOW_B = {"hwnd": 1002, "title": "Other", "rect": (600, 100, 400, 300)}

_s = get_settings()
DWELL_TIME_S = _s.intent.dwell_time_s
READY_TIMEOUT_S = _s.intent.ready_timeout_s
COOLDOWN_S = _s.intent.cooldown_s
CONF_THR = _s.gesture.confidence_threshold


def _machine() -> IntentStateMachine:
    return IntentStateMachine()


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


def test_initial_state_is_idle():
    m = _machine()
    assert m.state == "IDLE"


def test_idle_to_targeting_on_window():
    m = _machine()
    m.update(gaze_point=(200, 200), target_window=WINDOW_A, gesture_id=None, gesture_confidence=0.0)
    assert m.state == "TARGETING"


def test_targeting_to_idle_on_window_change():
    m = _machine()
    m.update((200, 200), WINDOW_A, None, 0.0)
    assert m.state == "TARGETING"
    m.update((200, 200), WINDOW_B, None, 0.0)
    assert m.state == "IDLE"


def test_targeting_to_ready_after_dwell():
    m = _machine()
    m.update((200, 200), WINDOW_A, None, 0.0)
    # Simulate elapsed dwell by backdating _dwell_start.
    m._dwell_start -= DWELL_TIME_S + 0.05
    m.update((200, 200), WINDOW_A, None, 0.0)
    assert m.state == "READY"


def test_ready_to_acting_on_gesture():
    m = _machine()
    m.state = "READY"
    m._target_window = WINDOW_A
    m._ready_start = time.monotonic()

    action = m.update((200, 200), WINDOW_A, "CLOSE_SIGN", CONF_THR + 0.01)
    assert m.state == "COOLDOWN"
    assert action is not None
    assert action["type"] == "CLOSE"


def test_ready_timeout_returns_to_idle():
    m = _machine()
    m.state = "READY"
    m._target_window = WINDOW_A
    m._ready_start = time.monotonic() - READY_TIMEOUT_S - 1.0

    m.update((200, 200), WINDOW_A, None, 0.0)
    assert m.state == "IDLE"


# ---------------------------------------------------------------------------
# DRAG / RESIZE lifecycle
# ---------------------------------------------------------------------------


def test_drag_lifecycle():
    m = _machine()
    m.state = "READY"
    m._target_window = WINDOW_A
    m._ready_start = time.monotonic()
    hand = (200.0, 200.0)

    action = m.update((200, 200), WINDOW_A, "GRAB", CONF_THR + 0.01, hand_position=hand)
    assert m.state == "ACTING"
    assert action is not None
    assert action["data"]["phase"] == "start"

    hand2 = (250.0, 220.0)
    action = m.update((200, 200), WINDOW_A, "GRAB", CONF_THR + 0.01, hand_position=hand2)
    assert action is not None
    assert action["data"]["phase"] == "move"
    assert action["data"]["delta"] == (50.0, 20.0)

    action = m.update((200, 200), WINDOW_A, "RELEASE", CONF_THR + 0.01)
    assert m.state == "COOLDOWN"
    assert action is not None
    assert action["data"]["phase"] == "end"


def test_resize_lifecycle():
    m = _machine()
    m.state = "READY"
    m._target_window = WINDOW_A
    m._ready_start = time.monotonic()
    hand = (200.0, 200.0)

    action = m.update((200, 200), WINDOW_A, "PINCH", CONF_THR + 0.01, hand_position=hand)
    assert m.state == "ACTING"
    assert action is not None
    assert action["data"]["phase"] == "start"

    hand2 = (230.0, 250.0)
    action = m.update((200, 200), WINDOW_A, "PINCH", CONF_THR + 0.01, hand_position=hand2)
    assert action is not None
    assert action["data"]["phase"] == "move"
    assert action["data"]["delta"] == (30.0, 50.0)


# ---------------------------------------------------------------------------
# Double-pinch / cooldown
# ---------------------------------------------------------------------------


def test_double_pinch_closes_app():
    m = _machine()
    # Simulate first pinch already ended.
    m._pinch_active = False
    m._last_pinch_end_time = time.monotonic() - 0.2  # 0.2 s ago (within window)

    action = m.update((200, 200), WINDOW_A, "PINCH", CONF_THR + 0.01)
    assert action is not None
    assert action["type"] == "CLOSE_APP"


def test_double_pinch_debounce_requires_n_frames():
    """Pinch release must persist N consecutive non-PINCH frames before counting."""
    m = _machine()
    cfg = get_settings().gesture
    n = cfg.pinch_release_frames

    # Start a pinch.
    m.update((200, 200), WINDOW_A, "PINCH", CONF_THR + 0.01)
    assert m._pinch_active is True

    # Feed n-1 non-PINCH frames — should NOT yet register release.
    for _ in range(n - 1):
        m.update((200, 200), WINDOW_A, None, 0.0)
    assert m._pinch_active is True  # not yet released
    assert m._last_pinch_end_time is None

    # One more frame → release confirmed.
    m.update((200, 200), WINDOW_A, None, 0.0)
    assert m._pinch_active is False
    assert m._last_pinch_end_time is not None


def test_cooldown_returns_to_idle():
    m = _machine()
    m.state = "COOLDOWN"
    m._cooldown_start = time.monotonic() - COOLDOWN_S - 0.05
    m.update(None, None, None, 0.0)
    assert m.state == "IDLE"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_action_without_window():
    m = _machine()
    action = m.update(None, None, None, 0.0)
    assert action is None
    assert m.state == "IDLE"


def test_low_confidence_gesture_ignored():
    m = _machine()
    m.state = "READY"
    m._target_window = WINDOW_A
    m._ready_start = time.monotonic()

    action = m.update((200, 200), WINDOW_A, "CLOSE_SIGN", 0.1)  # below threshold
    assert action is None
    assert m.state == "READY"
