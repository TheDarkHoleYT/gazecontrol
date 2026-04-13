"""Tests for CaptureStage."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from gazecontrol.capture.frame_preprocessor import FrameQuality
from gazecontrol.pipeline.context import FrameContext


def _make_stage():
    """Build a CaptureStage with grabber and preprocessor mocked."""
    from gazecontrol.pipeline.capture_stage import CaptureStage

    stage = object.__new__(CaptureStage)
    stage.grabber = MagicMock()
    stage._preprocessor = MagicMock()

    quality_mock = MagicMock(spec=FrameQuality)
    quality_mock.is_usable = True
    quality_mock.laplacian_var = 100.0
    quality_mock.brightness_mean = 128.0

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    stage.grabber.read_bgr.return_value = (True, fake_frame)
    stage._preprocessor.process.return_value = (fake_frame, quality_mock)
    return stage


def test_capture_ok_on_good_frame():
    stage = _make_stage()
    ctx = stage.process(FrameContext())
    assert ctx.capture_ok is True
    assert ctx.frame_bgr is not None
    assert ctx.frame_rgb is not None


def test_capture_fail_when_no_frame():
    from gazecontrol.pipeline.capture_stage import CaptureStage

    stage = object.__new__(CaptureStage)
    stage.grabber = MagicMock()
    stage._preprocessor = MagicMock()
    stage.grabber.read_bgr.return_value = (False, None)

    ctx = stage.process(FrameContext())
    assert ctx.capture_ok is False
    assert ctx.frame_bgr is None


def test_rgb_and_bgr_same_shape():
    stage = _make_stage()
    ctx = stage.process(FrameContext())
    assert ctx.frame_bgr.shape == ctx.frame_rgb.shape


def test_quality_set():
    stage = _make_stage()
    ctx = stage.process(FrameContext())
    assert ctx.quality is not None
