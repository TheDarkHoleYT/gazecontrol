"""Tests for AppSettings — validators, env overrides, nested models."""

from __future__ import annotations

import pytest

from gazecontrol.settings import (
    AppSettings,
    GestureLabelsSettings,
    InteractionSettings,
    LoggingSettings,
    get_settings,
    reset_settings,
)

# ---------------------------------------------------------------------------
# LoggingSettings defaults
# ---------------------------------------------------------------------------


def test_logging_settings_defaults():
    ls = LoggingSettings()
    assert ls.level == "INFO"
    assert ls.format == "text"
    assert ls.rotation_mb == 5
    assert ls.backup_count == 5


def test_logging_settings_invalid_level():
    with pytest.raises(Exception):
        LoggingSettings(level="TRACE")  # type: ignore[arg-type]


def test_logging_settings_invalid_format():
    with pytest.raises(Exception):
        LoggingSettings(format="xml")  # type: ignore[arg-type]


def test_logging_settings_rotation_bounds():
    with pytest.raises(Exception):
        LoggingSettings(rotation_mb=0)
    with pytest.raises(Exception):
        LoggingSettings(rotation_mb=101)


# ---------------------------------------------------------------------------
# GestureLabelsSettings defaults
# ---------------------------------------------------------------------------


def test_gesture_labels_defaults():
    gls = GestureLabelsSettings()
    assert "PINCH" in gls.labels
    assert len(gls.labels) > 0


# ---------------------------------------------------------------------------
# InteractionSettings defaults and validation
# ---------------------------------------------------------------------------


def test_interaction_settings_defaults():
    s = InteractionSettings()
    assert s.tap_ms == 220
    assert s.hold_ms == 280
    assert s.grip_ratio == pytest.approx(0.18)
    assert s.cooldown_ms == 120


def test_interaction_settings_custom():
    s = InteractionSettings(tap_ms=150, hold_ms=300, grip_ratio=0.20)
    assert s.tap_ms == 150
    assert s.hold_ms == 300


# ---------------------------------------------------------------------------
# AppSettings structure
# ---------------------------------------------------------------------------


def test_app_settings_has_logging():
    s = AppSettings()
    assert isinstance(s.logging, LoggingSettings)


def test_app_settings_has_gesture_labels():
    s = AppSettings()
    assert isinstance(s.gesture_labels, GestureLabelsSettings)


def test_app_settings_has_interaction():
    s = AppSettings()
    assert isinstance(s.interaction, InteractionSettings)


def test_app_settings_has_gaze_eyehand_fields():
    """Gaze settings re-introduced for the EYE_HAND mode."""
    s = AppSettings()
    assert hasattr(s, "gaze")
    assert hasattr(s, "fusion")
    assert hasattr(s, "runtime")


# ---------------------------------------------------------------------------
# Env-var overrides (monkeypatch)
# ---------------------------------------------------------------------------


def test_env_override_logging_level(monkeypatch):
    monkeypatch.setenv("GAZECONTROL_LOGGING__LEVEL", "DEBUG")
    reset_settings()
    s = get_settings()
    assert s.logging.level == "DEBUG"
    reset_settings()


def test_env_override_logging_format(monkeypatch):
    monkeypatch.setenv("GAZECONTROL_LOGGING__FORMAT", "json")
    reset_settings()
    s = get_settings()
    assert s.logging.format == "json"
    reset_settings()


def test_env_override_camera_index(monkeypatch):
    monkeypatch.setenv("GAZECONTROL_CAMERA__INDEX", "2")
    reset_settings()
    s = get_settings()
    assert s.camera.index == 2
    reset_settings()


def test_env_override_interaction_tap_ms(monkeypatch):
    monkeypatch.setenv("GAZECONTROL_INTERACTION__TAP_MS", "180")
    reset_settings()
    s = get_settings()
    assert s.interaction.tap_ms == 180
    reset_settings()
