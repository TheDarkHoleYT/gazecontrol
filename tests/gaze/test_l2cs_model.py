"""Tests for L2CSModel — disabled gracefully when model absent."""
from __future__ import annotations

import numpy as np
import pytest


def test_l2cs_raises_file_not_found_when_model_missing(tmp_path):
    from gazecontrol.gaze.l2cs_model import L2CSModel

    missing = str(tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError):
        L2CSModel(missing)


def test_l2cs_predict_returns_none_when_session_none():
    """If _session was never set (e.g. mock bypass), predict returns None."""
    from gazecontrol.gaze.l2cs_model import L2CSModel

    # Bypass constructor entirely to avoid file check.
    model = object.__new__(L2CSModel)
    model._session = None
    model._input_name = None

    dummy_crop = np.zeros((1, 3, 224, 224), dtype=np.float32)
    assert model.predict(dummy_crop) is None


def test_l2cs_is_loaded_false_when_session_none():
    from gazecontrol.gaze.l2cs_model import L2CSModel

    model = object.__new__(L2CSModel)
    model._session = None
    assert model.is_loaded is False


def test_l2cs_bins_to_angle_softmax():
    """_bins_to_angle should return a weighted average of bin centres."""
    from gazecontrol.gaze.l2cs_model import L2CSModel

    model = object.__new__(L2CSModel)
    model._session = None

    # All logits equal → angle should be near 0 (symmetric bins).
    logits = np.zeros(90, dtype=np.float32)
    angle = model._bins_to_angle(logits)
    assert abs(angle) < 2.0  # float32 summation may drift slightly from zero
