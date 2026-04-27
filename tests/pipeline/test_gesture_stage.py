"""Tests for GestureStage (hand-only, fingertip-mapper aware)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gazecontrol.gesture.feature_extractor import FeatureSet
from gazecontrol.gesture.fingertip_mapper import FingertipMapper, VirtualDesktop
from gazecontrol.gesture.pinch_tracker import PinchEvent
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.gesture_stage import GestureStage


def _make_mapper() -> FingertipMapper:
    return FingertipMapper(VirtualDesktop(left=0, top=0, width=1920, height=1080))


def _make_feat(**kwargs) -> FeatureSet:
    """Create a FeatureSet with sane defaults for gesture stage tests."""
    defaults = dict(
        finger_states=[1, 1, 1, 1, 1],
        finger_angles=[0.0] * 5,
        palm_direction=0.0,
        hand_velocity_x=0.0,
        hand_velocity_y=0.0,
        thumb_index_distance=0.5,
        wrist_x=0.5,
        wrist_y=0.5,
        thumb_dir_y=0.0,
    )
    defaults.update(kwargs)
    return FeatureSet(**defaults)


def _make_stage():
    stage = object.__new__(GestureStage)
    stage._mapper = _make_mapper()
    stage._settings = None
    stage._hand_detector = MagicMock()
    stage._feature_extractor = MagicMock()
    stage._rule_classifier = MagicMock()
    stage._mlp_classifier = None
    stage._tcn_classifier = None
    stage._fusion = None
    stage._pinch_tracker = MagicMock()
    stage._pinch_tracker.update.return_value = PinchEvent.NONE
    return stage


def _fake_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _fake_hand_result(lm_x=0.5, lm_y=0.5):
    """Fake HandLandmarkerResult with lm[8] at (lm_x, lm_y)."""
    result = MagicMock()
    result.multi_hand_landmarks = [MagicMock()]
    landmarks = [MagicMock() for _ in range(21)]
    for lm in landmarks:
        lm.x = 0.5
        lm.y = 0.5
    landmarks[8].x = lm_x
    landmarks[8].y = lm_y
    result.multi_hand_landmarks[0].landmark = landmarks
    return result


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
    assert result.fingertip_screen is None
    assert result.pinch_event == PinchEvent.NONE


def test_fingertip_screen_computed():
    stage = _make_stage()
    hand_result = _fake_hand_result(lm_x=0.5, lm_y=0.5)
    stage._hand_detector.process.return_value = hand_result
    feat = _make_feat()
    stage._feature_extractor.extract.return_value = feat
    stage._rule_classifier.classify.return_value = (None, 0.0)
    ctx = FrameContext(capture_ok=True, frame_rgb=_fake_frame())
    result = stage.process(ctx)
    # lm[8].x=0.5, lm[8].y=0.5 → screen (960, 540)
    assert result.fingertip_screen == (960, 540)


def test_gesture_label_set():
    stage = _make_stage()
    hand_result = _fake_hand_result()
    stage._hand_detector.process.return_value = hand_result
    feat = _make_feat()
    stage._feature_extractor.extract.return_value = feat
    stage._rule_classifier.classify.return_value = ("PINCH", 0.92)
    ctx = FrameContext(capture_ok=True, frame_rgb=_fake_frame())
    result = stage.process(ctx)
    assert result.gesture_label == "PINCH"
    assert result.gesture_confidence == pytest.approx(0.92)


def test_scroll_delta_detected():
    stage = _make_stage()
    hand_result = _fake_hand_result()
    stage._hand_detector.process.return_value = hand_result
    # index + middle extended, others closed, scrolling up
    feat = _make_feat(finger_states=[0, 1, 1, 0, 0], hand_velocity_y=-300.0)
    stage._feature_extractor.extract.return_value = feat
    stage._rule_classifier.classify.return_value = (None, 0.0)
    ctx = FrameContext(capture_ok=True, frame_rgb=_fake_frame())
    result = stage.process(ctx)
    assert result.two_finger_scroll_delta == 120  # scroll up


def test_exception_does_not_propagate():
    stage = _make_stage()
    stage._hand_detector.process.side_effect = RuntimeError("camera error")
    ctx = FrameContext(capture_ok=True, frame_rgb=_fake_frame())
    result = stage.process(ctx)
    assert result is not None


def test_init_sets_mapper():
    mapper = _make_mapper()
    stage = GestureStage(fingertip_mapper=mapper)
    assert stage._mapper is mapper
    assert stage._hand_detector is None


def test_start_allocates_resources():
    mapper = _make_mapper()
    with (
        patch("gazecontrol.gesture.hand_detector.HandDetector", return_value=MagicMock()),
        patch("gazecontrol.gesture.mlp_classifier.MLPClassifier", side_effect=ImportError),
    ):
        stage = GestureStage(fingertip_mapper=mapper)
        ok = stage.start()
    assert ok is True
    assert stage._hand_detector is not None
    assert stage._rule_classifier is not None
    assert stage._pinch_tracker is not None
    assert stage._mlp_classifier is None


def test_stop_releases_resources():
    stage = _make_stage()
    mock_det = MagicMock()
    stage._hand_detector = mock_det
    stage.stop()
    mock_det.close.assert_called_once()
    assert stage._hand_detector is None
    assert stage._pinch_tracker is None
