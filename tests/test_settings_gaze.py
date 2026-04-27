"""Tests for the new GazeSettings/FusionSettings/RuntimeSettings sections."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gazecontrol.settings import (
    AppSettings,
    DriftCorrectorSettings,
    FusionSettings,
    GazeSettings,
    InputMode,
    RuntimeSettings,
)


def test_defaults():
    s = AppSettings()
    assert s.runtime.input_mode == InputMode.HAND_ONLY
    assert s.runtime.show_mode_selector is True
    assert s.gaze.backend == "ensemble"
    assert s.fusion.gaze_assisted_click is False


def test_runtime_input_mode_validation():
    rs = RuntimeSettings(input_mode="eye-hand")
    assert rs.input_mode == InputMode.EYE_HAND
    with pytest.raises(ValidationError):
        RuntimeSettings(input_mode="brainwave")


def test_gaze_backend_choice_validation():
    g = GazeSettings(backend="l2cs")
    assert g.backend == "l2cs"
    with pytest.raises(ValidationError):
        GazeSettings(backend="random")


def test_fusion_threshold_bounds():
    with pytest.raises(ValidationError):
        FusionSettings(hand_confidence_threshold=1.5)
    with pytest.raises(ValidationError):
        FusionSettings(divergence_threshold_px=-1)


def test_drift_corrector_defaults():
    d = DriftCorrectorSettings()
    assert d.enabled is True
    assert 0 < d.implicit_alpha <= 1


def test_env_override_input_mode(monkeypatch):
    monkeypatch.setenv("GAZECONTROL_RUNTIME__INPUT_MODE", "eye-hand")
    s = AppSettings()
    assert s.runtime.input_mode == InputMode.EYE_HAND


def test_input_mode_string_value():
    assert str(InputMode.HAND_ONLY) == "hand"
    assert str(InputMode.EYE_HAND) == "eye-hand"
