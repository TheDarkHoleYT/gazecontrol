"""Tests for ActionStage (hand-only interaction dispatcher)."""

from __future__ import annotations

from unittest.mock import MagicMock

from gazecontrol.interaction.types import HoveredWindow, Interaction, InteractionKind
from gazecontrol.pipeline.action_stage import ActionStage
from gazecontrol.pipeline.context import FrameContext


def _make_stage(**kwargs) -> ActionStage:
    wm = MagicMock()
    return ActionStage(window_manager=wm, **kwargs)


def _ctx_with(kind: InteractionKind, window=None, point=(100, 100), data=None) -> FrameContext:
    ctx = FrameContext()
    ctx.interaction = Interaction(kind=kind, window=window, point=point, data=data or {})
    return ctx


# ---------------------------------------------------------------------------
# No-op when interaction is absent
# ---------------------------------------------------------------------------


def test_no_interaction_is_noop():
    stage = _make_stage()
    ctx = FrameContext()
    result = stage.process(ctx)
    stage._wm.click_at.assert_not_called()
    assert result is ctx


# ---------------------------------------------------------------------------
# CLICK
# ---------------------------------------------------------------------------


def test_click_calls_click_at():
    stage = _make_stage()
    ctx = _ctx_with(InteractionKind.CLICK, point=(200, 300))
    stage.process(ctx)
    stage._wm.click_at.assert_called_once_with(200, 300)


# ---------------------------------------------------------------------------
# DRAG_UPDATE
# ---------------------------------------------------------------------------


def test_drag_update_calls_move_window():
    stage = _make_stage()
    hw = HoveredWindow(hwnd=42, rect=(0, 0, 800, 600))
    ctx = _ctx_with(InteractionKind.DRAG_UPDATE, window=hw, data={"new_x": 50, "new_y": 100})
    stage.process(ctx)
    stage._wm.move_window.assert_called_once_with(42, 50, 100)


def test_drag_update_no_window_is_noop():
    stage = _make_stage()
    ctx = _ctx_with(InteractionKind.DRAG_UPDATE, window=None, data={"new_x": 50, "new_y": 100})
    stage.process(ctx)
    stage._wm.move_window.assert_not_called()


# ---------------------------------------------------------------------------
# RESIZE_UPDATE
# ---------------------------------------------------------------------------


def test_resize_update_calls_resize_window():
    stage = _make_stage()
    hw = HoveredWindow(hwnd=7, rect=(10, 20, 800, 600))
    ctx = _ctx_with(
        InteractionKind.RESIZE_UPDATE,
        window=hw,
        data={"new_x": 10, "new_y": 20, "new_w": 900, "new_h": 700},
    )
    stage.process(ctx)
    stage._wm.resize_window.assert_called_once_with(7, 10, 20, 900, 700)


# ---------------------------------------------------------------------------
# SCROLL
# ---------------------------------------------------------------------------


def test_scroll_up_calls_scroll_at():
    stage = _make_stage()
    ctx = _ctx_with(InteractionKind.SCROLL_UP, point=(400, 300))
    stage.process(ctx)
    stage._wm.scroll_at.assert_called_once_with(400, 300, delta=120)


def test_scroll_down_calls_scroll_at():
    stage = _make_stage()
    ctx = _ctx_with(InteractionKind.SCROLL_DOWN, point=(400, 300))
    stage.process(ctx)
    stage._wm.scroll_at.assert_called_once_with(400, 300, delta=-120)


# ---------------------------------------------------------------------------
# TOGGLE_LAUNCHER
# ---------------------------------------------------------------------------


def test_toggle_launcher_calls_bridge():
    bridge = MagicMock()
    stage = _make_stage(overlay_bridge=bridge)
    ctx = _ctx_with(InteractionKind.TOGGLE_LAUNCHER)
    stage.process(ctx)
    bridge.assert_called_once()


def test_toggle_launcher_no_bridge_is_noop():
    stage = _make_stage()
    ctx = _ctx_with(InteractionKind.TOGGLE_LAUNCHER)
    stage.process(ctx)  # must not raise


# ---------------------------------------------------------------------------
# on_stop
# ---------------------------------------------------------------------------


def test_on_stop_not_called_for_normal_interactions():
    on_stop = MagicMock()
    stage = _make_stage(on_stop=on_stop)
    ctx = _ctx_with(InteractionKind.CLICK)
    stage.process(ctx)
    on_stop.assert_not_called()


# ---------------------------------------------------------------------------
# PipelineStage Protocol
# ---------------------------------------------------------------------------


def test_start_returns_true():
    stage = _make_stage()
    assert stage.start() is True


def test_stop_is_idempotent():
    stage = _make_stage()
    stage.stop()
    stage.stop()


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------


def test_wm_exception_does_not_propagate():
    stage = _make_stage()
    stage._wm.click_at.side_effect = RuntimeError("wm crash")
    ctx = _ctx_with(InteractionKind.CLICK)
    result = stage.process(ctx)
    assert result is not None
