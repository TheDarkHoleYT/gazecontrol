"""Tests for InteractionStage (replaces old intent_stage tests)."""

from __future__ import annotations

from unittest.mock import MagicMock

from gazecontrol.gesture.pinch_tracker import PinchEvent
from gazecontrol.interaction.types import Interaction, InteractionKind
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.interaction_stage import InteractionStage


def _make_stage():
    stage = object.__new__(InteractionStage)
    stage._fsm = MagicMock()
    stage._hit_tester = MagicMock()
    stage._fsm.state = "IDLE"
    stage._fsm.update.return_value = None
    stage._hit_tester.at.return_value = None
    return stage


def test_state_property():
    stage = _make_stage()
    assert stage.state == "IDLE"


def test_process_no_fingertip_skips_hit_tester():
    stage = _make_stage()
    ctx = FrameContext(fingertip_screen=None, pinch_event=PinchEvent.NONE)
    stage.process(ctx)
    stage._hit_tester.at.assert_not_called()


def test_process_with_fingertip_calls_hit_tester():
    stage = _make_stage()
    ctx = FrameContext(fingertip_screen=(500, 300), pinch_event=PinchEvent.NONE)
    stage.process(ctx)
    stage._hit_tester.at.assert_called_once_with((500, 300))


def test_process_sets_hovered_window():
    stage = _make_stage()
    from gazecontrol.interaction.types import HoveredWindow

    hw = HoveredWindow(hwnd=42, rect=(0, 0, 800, 600), title="Test")
    stage._hit_tester.at.return_value = hw
    ctx = FrameContext(fingertip_screen=(100, 100), pinch_event=PinchEvent.NONE)
    result = stage.process(ctx)
    assert result.hovered_window is hw


def test_process_sets_interaction():
    stage = _make_stage()
    interaction = Interaction(kind=InteractionKind.CLICK, point=(200, 300))
    stage._fsm.update.return_value = interaction
    ctx = FrameContext(fingertip_screen=(200, 300), pinch_event=PinchEvent.DOWN)
    result = stage.process(ctx)
    assert result.interaction is interaction


def test_start_returns_true():
    stage = _make_stage()
    assert stage.start() is True


def test_stop_is_noop():
    stage = _make_stage()
    stage.stop()  # must not raise


def test_fsm_exception_does_not_propagate():
    stage = _make_stage()
    stage._fsm.update.side_effect = RuntimeError("crash")
    ctx = FrameContext(fingertip_screen=(0, 0), pinch_event=PinchEvent.NONE)
    result = stage.process(ctx)
    assert result is not None
    assert result.interaction is None
