"""Tests for FrameContext dataclass."""
from __future__ import annotations

import time

import numpy as np

from gazecontrol.pipeline.context import FrameContext


def test_default_fields():
    ctx = FrameContext()
    assert ctx.frame_bgr is None
    assert ctx.capture_ok is False
    assert ctx.blink is False
    assert ctx.gesture_label is None
    assert ctx.action is None


def test_set_frame_fields():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx = FrameContext(frame_bgr=frame, capture_ok=True)
    assert ctx.capture_ok is True
    assert ctx.frame_bgr is frame


def test_t0_field():
    t = time.monotonic()
    ctx = FrameContext(t0=t)
    assert ctx.t0 == t


def test_gaze_fields():
    ctx = FrameContext(
        gaze_raw=(100.0, 200.0),
        gaze_filtered=(101.0, 201.0),
        gaze_corrected=(102.0, 202.0),
        gaze_point=(102, 202),
    )
    assert ctx.gaze_raw == (100.0, 200.0)
    assert ctx.gaze_point == (102, 202)


def test_gesture_confidence_default():
    ctx = FrameContext()
    assert ctx.gesture_confidence == 0.0
