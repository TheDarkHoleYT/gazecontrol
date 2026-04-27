"""Tests for FrameContext dataclass."""

from __future__ import annotations

import time

import numpy as np

from gazecontrol.pipeline.context import FrameContext


def test_default_fields():
    ctx = FrameContext()
    assert ctx.frame_bgr is None
    assert ctx.capture_ok is False
    assert ctx.gesture_label is None
    assert ctx.interaction is None
    assert ctx.fingertip_screen is None
    assert ctx.two_finger_scroll_delta == 0


def test_set_frame_fields():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx = FrameContext(frame_bgr=frame, capture_ok=True)
    assert ctx.capture_ok is True
    assert ctx.frame_bgr is frame


def test_t0_field():
    t = time.monotonic()
    ctx = FrameContext(t0=t)
    assert ctx.t0 == t


def test_gesture_confidence_default():
    ctx = FrameContext()
    assert ctx.gesture_confidence == 0.0


def test_fingertip_screen_settable():
    ctx = FrameContext()
    ctx.fingertip_screen = (960, 540)
    assert ctx.fingertip_screen == (960, 540)


def test_hovered_window_settable():
    from gazecontrol.interaction.types import HoveredWindow

    ctx = FrameContext()
    hw = HoveredWindow(hwnd=42, rect=(0, 0, 800, 600), title="Test")
    ctx.hovered_window = hw
    assert ctx.hovered_window is hw


def test_interaction_settable():
    from gazecontrol.interaction.types import Interaction, InteractionKind

    ctx = FrameContext()
    i = Interaction(kind=InteractionKind.CLICK, point=(100, 200))
    ctx.interaction = i
    assert ctx.interaction is i


def test_no_gaze_fields():
    """FrameContext must NOT have legacy gaze_* fields."""
    ctx = FrameContext()
    assert not hasattr(ctx, "gaze_point")
    assert not hasattr(ctx, "gaze_raw")
    assert not hasattr(ctx, "blink")
    assert not hasattr(ctx, "fixation_event")
