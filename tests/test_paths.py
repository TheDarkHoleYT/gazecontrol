"""Tests for gazecontrol.paths — path factory methods."""

from __future__ import annotations

from pathlib import Path

from gazecontrol.paths import Paths


def test_profiles_returns_path(tmp_path):
    p = Paths.profiles(override=tmp_path / "profiles")
    assert isinstance(p, Path)
    assert p.exists()


def test_profiles_override(tmp_path):
    override = tmp_path / "my_profiles"
    p = Paths.profiles(override=override)
    assert p == override
    assert p.exists()


def test_log_file_returns_path(tmp_path):
    p = Paths.log_file(override=tmp_path / "test.log")
    assert isinstance(p, Path)
    assert p.name == "test.log"


def test_log_file_override(tmp_path):
    target = tmp_path / "custom.log"
    p = Paths.log_file(override=target)
    assert p == target


def test_models_override(tmp_path):
    override = tmp_path / "models"
    override.mkdir()
    p = Paths.models(override=override)
    assert p == override


def test_gesture_mlp_model_name():
    p = Paths.gesture_mlp_model()
    assert p.name == "gesture_mlp.onnx"


def test_hand_landmarker_name():
    p = Paths.hand_landmarker()
    assert p.name == "hand_landmarker.task"


def test_launcher_config_name():
    p = Paths.launcher_config()
    assert p.name == "launcher.toml"


def test_launcher_config_override(tmp_path):
    override = tmp_path / "my_launcher.toml"
    p = Paths.launcher_config(override=override)
    assert p == override
