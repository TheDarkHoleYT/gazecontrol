"""Tests for IntentStage."""
from __future__ import annotations

from unittest.mock import MagicMock

from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.intent_stage import IntentStage


def _make_stage():
    stage = object.__new__(IntentStage)
    stage._state_machine = MagicMock()
    stage._window_selector = MagicMock()
    stage._state_machine.state = "IDLE"
    stage._state_machine.update.return_value = None
    stage._window_selector.find_window.return_value = {"hwnd": 42}
    return stage


def test_process_no_gaze_skips_window_selector():
    stage = _make_stage()
    ctx = FrameContext(gaze_point=None)
    stage.process(ctx)
    stage._window_selector.find_window.assert_not_called()


def test_process_with_gaze_calls_window_selector():
    stage = _make_stage()
    ctx = FrameContext(gaze_point=(500, 300))
    stage.process(ctx)
    stage._window_selector.find_window.assert_called_once_with((500, 300))


def test_process_sets_target_window():
    stage = _make_stage()
    ctx = FrameContext(gaze_point=(500, 300))
    result = stage.process(ctx)
    assert result.target_window == {"hwnd": 42}


def test_state_property():
    stage = _make_stage()
    assert stage.state == "IDLE"


def test_process_passes_gesture_to_fsm():
    stage = _make_stage()
    ctx = FrameContext(
        gaze_point=(100, 100),
        gesture_label="PINCH",
        gesture_confidence=0.95,
        hand_position=(400.0, 300.0),
    )
    stage.process(ctx)
    call_kwargs = stage._state_machine.update.call_args[1]
    assert call_kwargs["gesture_id"] == "PINCH"
    assert call_kwargs["gesture_confidence"] == 0.95
