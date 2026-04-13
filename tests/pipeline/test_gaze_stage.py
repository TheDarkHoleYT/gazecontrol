"""Tests for GazeStage."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np

from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.gaze_stage import GazeStage


def _make_stage() -> GazeStage:
    """Build a GazeStage with all sub-components mocked."""
    with patch("gazecontrol.pipeline.gaze_stage.Paths") as mock_paths:
        mock_paths.l2cs_model.return_value = MagicMock(exists=lambda: False)
        stage = GazeStage(screen_w=1920, screen_h=1080)

    # Replace internals with mocks.
    stage._filter_x = MagicMock()
    stage._filter_y = MagicMock()
    stage._filter_x.filter.return_value = 500.0
    stage._filter_y.filter.return_value = 300.0
    stage._drift_corrector = MagicMock()
    stage._drift_corrector.correct.return_value = (500.0, 300.0)
    stage._fixation_detector = MagicMock()
    event = MagicMock()
    event.type = "fixation"
    event.centroid = (500.0, 300.0)
    stage._fixation_detector.update.return_value = event
    stage._l2cs_enabled = False
    return stage


def _base_ctx() -> FrameContext:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return FrameContext(
        frame_bgr=frame,
        frame_rgb=frame,
        capture_ok=True,
        t0=time.monotonic(),
    )


def test_skip_when_not_calibrated():
    stage = _make_stage()
    stage.is_calibrated = False
    ctx = _base_ctx()
    result = stage.process(ctx)
    assert result.gaze_point is None


def test_skip_when_capture_not_ok():
    stage = _make_stage()
    stage.is_calibrated = True
    stage.estimator = MagicMock()
    ctx = FrameContext(capture_ok=False)
    stage.process(ctx)
    stage.estimator.extract_features.assert_not_called()


def test_blink_holds_last_gaze():
    stage = _make_stage()
    stage.is_calibrated = True
    stage.estimator = MagicMock()
    stage.estimator.extract_features.return_value = (None, True)
    stage._last_valid_gaze = (400, 250)
    stage._blink_start = None

    ctx = _base_ctx()
    ctx.t0 = time.monotonic()
    result = stage.process(ctx)
    assert result.blink is True
    assert result.gaze_point == (400, 250)


def test_full_pipeline_sets_gaze_point():
    stage = _make_stage()
    stage.is_calibrated = True

    fake_features = np.zeros(10)
    stage.estimator = MagicMock()
    stage.estimator.extract_features.return_value = (fake_features, False)
    stage.estimator.predict.return_value = [[500.0, 300.0]]

    ctx = _base_ctx()
    result = stage.process(ctx)
    assert result.gaze_point is not None
    assert result.gaze_raw == (500.0, 300.0)
    assert result.gaze_filtered is not None
    assert result.gaze_corrected is not None


def test_exception_in_estimator_does_not_propagate():
    stage = _make_stage()
    stage.is_calibrated = True
    stage.estimator = MagicMock()
    stage.estimator.extract_features.side_effect = RuntimeError("mediapipe crash")

    ctx = _base_ctx()
    result = stage.process(ctx)
    # Should not raise; gaze_point remains None.
    assert result is not None


def test_saccade_uses_raw_coords():
    stage = _make_stage()
    stage.is_calibrated = True

    fake_features = np.zeros(10)
    stage.estimator = MagicMock()
    stage.estimator.extract_features.return_value = (fake_features, False)
    stage.estimator.predict.return_value = [[800.0, 600.0]]
    stage._filter_x.filter.return_value = 800.0
    stage._filter_y.filter.return_value = 600.0
    stage._drift_corrector.correct.return_value = (800.0, 600.0)

    event = MagicMock()
    event.type = "saccade"
    event.centroid = None
    stage._fixation_detector.update.return_value = event

    ctx = _base_ctx()
    result = stage.process(ctx)
    assert result.gaze_point == (800, 600)


def test_fixation_centroid_used():
    stage = _make_stage()
    stage.is_calibrated = True

    fake_features = np.zeros(10)
    stage.estimator = MagicMock()
    stage.estimator.extract_features.return_value = (fake_features, False)
    stage.estimator.predict.return_value = [[500.0, 300.0]]

    event = MagicMock()
    event.type = "fixation"
    event.centroid = (510.0, 305.0)
    stage._fixation_detector.update.return_value = event

    ctx = _base_ctx()
    result = stage.process(ctx)
    assert result.gaze_point == (510, 305)
