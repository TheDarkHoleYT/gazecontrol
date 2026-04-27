"""Tests for PinchTracker — hysteresis-based pinch event detector."""

from __future__ import annotations

import pytest

from gazecontrol.gesture.pinch_tracker import PinchEvent, PinchTracker


def test_down_event_when_aperture_below_threshold():
    t = PinchTracker(down_threshold=0.22, up_threshold=0.32)
    event = t.update(0.15)
    assert event == PinchEvent.DOWN


def test_hold_after_down():
    t = PinchTracker(down_threshold=0.22, up_threshold=0.32)
    t.update(0.15)  # DOWN
    event = t.update(0.18)  # HOLD
    assert event == PinchEvent.HOLD


def test_up_event_when_aperture_exceeds_up_threshold():
    t = PinchTracker(down_threshold=0.22, up_threshold=0.32)
    t.update(0.15)  # DOWN
    t.update(0.18)  # HOLD
    event = t.update(0.35)  # UP
    assert event == PinchEvent.UP


def test_none_when_not_pinching_and_above_down():
    t = PinchTracker(down_threshold=0.22, up_threshold=0.32)
    event = t.update(0.50)
    assert event == PinchEvent.NONE


def test_hysteresis_band_does_not_trigger_up_prematurely():
    t = PinchTracker(down_threshold=0.22, up_threshold=0.32)
    t.update(0.15)  # DOWN
    event = t.update(0.25)  # within band [0.22, 0.32] — still HOLD
    assert event == PinchEvent.HOLD


def test_hysteresis_band_does_not_trigger_down_prematurely():
    t = PinchTracker(down_threshold=0.22, up_threshold=0.32)
    event = t.update(0.25)  # above down_threshold → NONE
    assert event == PinchEvent.NONE


def test_is_pinching_after_down():
    t = PinchTracker()
    t.update(0.10)
    assert t.is_pinching is True


def test_is_not_pinching_after_up():
    t = PinchTracker()
    t.update(0.10)
    t.update(0.40)
    assert t.is_pinching is False


def test_reset_clears_state():
    t = PinchTracker()
    t.update(0.10)
    t.reset()
    assert t.is_pinching is False
    assert t.pinch_start_time is None
    event = t.update(0.50)
    assert event == PinchEvent.NONE


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        PinchTracker(down_threshold=0.30, up_threshold=0.20)


def test_sequence_down_hold_up_down_hold_up():
    t = PinchTracker(down_threshold=0.22, up_threshold=0.32)
    seq = [0.10, 0.15, 0.20, 0.35, 0.10, 0.20, 0.40]
    events = [t.update(a) for a in seq]
    assert events[0] == PinchEvent.DOWN
    assert events[1] == PinchEvent.HOLD
    assert events[2] == PinchEvent.HOLD
    assert events[3] == PinchEvent.UP
    assert events[4] == PinchEvent.DOWN
    assert events[6] == PinchEvent.UP
