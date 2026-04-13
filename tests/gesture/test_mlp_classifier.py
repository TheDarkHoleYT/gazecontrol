"""Tests for MLPClassifier — ONNX-backed gesture classifier."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gazecontrol.gesture.mlp_classifier import MLPClassifier, _features_to_vector


def _sample_features() -> dict:
    return {
        "finger_states": [1, 0, 0, 0, 0],
        "finger_angles": [10.0, 20.0, 15.0, 18.0, 22.0],
        "palm_direction": 0.8,
        "hand_velocity_x": 5.0,
        "hand_velocity_y": -3.0,
        "thumb_index_distance": 0.1,
        "wrist_x": 0.5,
        "wrist_y": 0.6,
    }


# ---------------------------------------------------------------------------
# _features_to_vector
# ---------------------------------------------------------------------------


def test_features_to_vector_shape():
    vec = _features_to_vector(_sample_features())
    # 5 finger_states + 5 finger_angles + 4 scalars (palm_dir, vel_x, vel_y, thumb_idx) = 14
    assert vec.shape == (1, 14)
    assert vec.dtype == np.float32


def test_features_to_vector_values():
    feat = _sample_features()
    vec = _features_to_vector(feat)
    # First element should be finger_states[0] = 1.0
    assert vec[0, 0] == 1.0


# ---------------------------------------------------------------------------
# MLPClassifier — disabled state (model not found)
# ---------------------------------------------------------------------------


def test_classifier_disabled_when_no_model(tmp_path):
    with patch("gazecontrol.gesture.mlp_classifier.Paths") as mock_paths:
        mock_paths.gesture_mlp_model.return_value = tmp_path / "no_such_model.onnx"
        clf = MLPClassifier()
    assert clf.is_loaded() is False


def test_classify_returns_none_when_not_loaded(tmp_path):
    with patch("gazecontrol.gesture.mlp_classifier.Paths") as mock_paths:
        mock_paths.gesture_mlp_model.return_value = tmp_path / "no_such_model.onnx"
        clf = MLPClassifier()
    label, conf = clf.classify(_sample_features())
    assert label is None
    assert conf == 0.0


def test_classify_none_input_returns_none(tmp_path):
    with patch("gazecontrol.gesture.mlp_classifier.Paths") as mock_paths:
        mock_paths.gesture_mlp_model.return_value = tmp_path / "no_such_model.onnx"
        clf = MLPClassifier()
    label, conf = clf.classify(None)
    assert label is None
    assert conf == 0.0


# ---------------------------------------------------------------------------
# MLPClassifier — mock ONNX session
# ---------------------------------------------------------------------------


def _make_loaded_clf() -> MLPClassifier:
    """Classifier with a mocked ONNX session."""
    clf = object.__new__(MLPClassifier)
    clf._loaded = True
    clf._input_name = "input"
    clf._session = MagicMock()
    return clf


def test_classify_with_mock_session_valid_output():
    clf = _make_loaded_clf()
    probas = {"PINCH": 0.9, "SWIPE_LEFT": 0.05, "SWIPE_RIGHT": 0.03, "MAXIMIZE": 0.02}
    clf._session.run.return_value = [None, [probas]]
    label, conf = clf.classify(_sample_features())
    assert label == "PINCH"
    assert conf == pytest.approx(0.9)


def test_classify_with_non_dict_probas_returns_none():
    clf = _make_loaded_clf()
    clf._session.run.return_value = [None, [np.array([0.9, 0.05, 0.03, 0.02])]]
    label, conf = clf.classify(_sample_features())
    assert label is None
    assert conf == 0.0


def test_classify_inference_error_returns_none():
    clf = _make_loaded_clf()
    clf._session.run.side_effect = RuntimeError("ONNX error")
    label, conf = clf.classify(_sample_features())
    assert label is None
    assert conf == 0.0


def test_load_failure_sets_not_loaded(tmp_path):
    clf = object.__new__(MLPClassifier)
    clf._session = None
    clf._input_name = None
    clf._loaded = False
    result = clf.load(str(tmp_path / "bad.onnx"))
    assert result is False
    assert clf.is_loaded() is False
