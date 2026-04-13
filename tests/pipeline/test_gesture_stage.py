"""Tests for GestureStage."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.gesture_stage import GestureStage


def _make_stage():
    stage = object.__new__(GestureStage)
    stage._screen_w = 1920
    stage._screen_h = 1080
    from gazecontrol.settings import get_settings
    stage._cfg = get_settings().gesture
    stage._hand_detector = MagicMock()
    stage._feature_extractor = MagicMock()
    stage._rule_classifier = MagicMock()
    stage._mlp_classifier = None
    stage._ts_ms = 0
    return stage


def _fake_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_skip_when_capture_not_ok():
    stage = _make_stage()
    ctx = FrameContext(capture_ok=False)
    result = stage.process(ctx)
    stage._hand_detector.process.assert_not_called()
    assert result.gesture_label is None


def test_no_hand_clears_gesture():
    stage = _make_stage()
    stage._hand_detector.process.return_value = None
    ctx = FrameContext(capture_ok=True, frame_rgb=_fake_frame())
    result = stage.process(ctx)
    assert result.gesture_label is None
    assert result.gesture_confidence == 0.0
    assert result.hand_position is None


def test_gesture_label_set():
    stage = _make_stage()

    fake_result = MagicMock()
    stage._hand_detector.process.return_value = fake_result

    fake_feat = {
        "wrist_x": 0.5, "wrist_y": 0.5,
        "finger_states": [1, 1, 1, 1, 1],
        "finger_angles": [0.0] * 5,
        "palm_direction": 0.0,
        "hand_velocity_x": 0.0,
        "hand_velocity_y": 0.0,
        "thumb_index_distance": 0.5,
    }
    stage._feature_extractor.extract.return_value = fake_feat
    stage._rule_classifier.classify.return_value = ("PINCH", 0.9)

    ctx = FrameContext(capture_ok=True, frame_rgb=_fake_frame())
    result = stage.process(ctx)
    assert result.gesture_label == "PINCH"
    assert result.gesture_confidence == 0.9


def test_exception_does_not_propagate():
    stage = _make_stage()
    stage._hand_detector.process.side_effect = RuntimeError("camera error")
    ctx = FrameContext(capture_ok=True, frame_rgb=_fake_frame())
    # Should not raise.
    result = stage.process(ctx)
    assert result is not None
