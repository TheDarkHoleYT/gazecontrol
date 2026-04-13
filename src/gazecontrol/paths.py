"""GazeControl path resolution.

All paths are resolved via ``platformdirs`` for user data/config/logs
so the package works correctly whether installed as a wheel (site-packages)
or run in-place from the source tree.

Usage::

    from gazecontrol.paths import Paths

    profile_dir = Paths.profiles() / "default"
    log_file = Paths.log_file()
    model = Paths.models() / "gesture_mlp.onnx"
"""
from __future__ import annotations

import importlib.resources
import os
from functools import lru_cache
from pathlib import Path

import platformdirs

APP_NAME = "gazecontrol"
APP_AUTHOR = "GazeControl"


@lru_cache(maxsize=None)
def _user_config_base() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME, APP_AUTHOR))


@lru_cache(maxsize=None)
def _user_log_base() -> Path:
    return Path(platformdirs.user_log_dir(APP_NAME, APP_AUTHOR))


@lru_cache(maxsize=None)
def _package_root() -> Path:
    """Root of the installed/editable package (contains src/ in dev mode)."""
    try:
        ref = importlib.resources.files("gazecontrol")
        pkg_path = Path(str(ref))  # editable: .../src/gazecontrol
        # go up to the project root (src/gazecontrol → src → project)
        return pkg_path.parent.parent
    except Exception:  # noqa: BLE001
        return Path.cwd()


class Paths:
    """Centralised path factory.

    All methods return ``Path`` objects. Directories are created on first
    access if they do not exist.
    """

    @staticmethod
    def profiles(override: str | os.PathLike[str] | None = None) -> Path:
        """Return the profiles directory, creating it if needed."""
        if override:
            path = Path(override)
        else:
            path = _user_config_base() / "profiles"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def log_file(override: str | os.PathLike[str] | None = None) -> Path:
        """Return the log file path, creating parent dirs if needed."""
        if override:
            path = Path(override)
        else:
            log_dir = _user_log_base()
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / "gazecontrol.log"
        return path

    @staticmethod
    def models(override: str | os.PathLike[str] | None = None) -> Path:
        """Return the models directory.

        Falls back to ``<project_root>/models`` for development installs,
        or ``<user_config_dir>/models`` for wheel installs.
        """
        if override:
            path = Path(override)
        else:
            dev_models = _package_root() / "models"
            if dev_models.exists():
                path = dev_models
            else:
                path = _user_config_base() / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def l2cs_model() -> Path:
        """Return the L2CS-Net ONNX model path."""
        return Paths.models() / "l2cs_net_gaze360.onnx"

    @staticmethod
    def gesture_mlp_model() -> Path:
        """Return the gesture MLP ONNX model path."""
        return Paths.models() / "gesture_mlp.onnx"

    @staticmethod
    def hand_landmarker() -> Path:
        """Return the MediaPipe hand landmarker task file path."""
        return Paths.models() / "hand_landmarker.task"

    @staticmethod
    def face_landmarker() -> Path:
        """Return the MediaPipe face landmarker task file path."""
        return Paths.models() / "face_landmarker.task"
