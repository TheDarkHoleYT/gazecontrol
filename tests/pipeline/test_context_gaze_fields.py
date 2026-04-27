"""Verify FrameContext exposes the new gaze + pointer fields with safe defaults."""

from __future__ import annotations

from gazecontrol.pipeline.context import FrameContext


def test_default_context_safe_for_hand_only():
    ctx = FrameContext()
    assert ctx.gaze_screen is None
    assert ctx.gaze_confidence == 0.0
    assert ctx.gaze_event is None
    assert ctx.gaze_blink is False
    assert ctx.face_present is False
    assert ctx.pointer_screen is None
    assert ctx.pointer_source == "hand"


def test_pointer_fields_assignable():
    ctx = FrameContext()
    ctx.pointer_screen = (10, 20)
    ctx.pointer_source = "gaze"
    assert ctx.pointer_screen == (10, 20)
    assert ctx.pointer_source == "gaze"
