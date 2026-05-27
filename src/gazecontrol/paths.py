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
from functools import cache
from pathlib import Path

import platformdirs

APP_NAME = "gazecontrol"
APP_AUTHOR = "GazeControl"


@cache
def _user_config_base() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME, APP_AUTHOR))


@cache
def _user_log_base() -> Path:
    return Path(platformdirs.user_log_dir(APP_NAME, APP_AUTHOR))


@cache
def _package_root() -> Path:
    """Root of the installed/editable package (contains src/ in dev mode)."""
    try:
        ref = importlib.resources.files("gazecontrol")
        pkg_path = Path(str(ref))  # editable: .../src/gazecontrol
        # go up to the project root (src/gazecontrol → src → project)
        return pkg_path.parent.parent
    except Exception:
        return Path.cwd()


class Paths:
    """Centralised path factory.

    All methods return ``Path`` objects. Directories are created on first
    access if they do not exist.
    """

    @staticmethod
    def profiles(override: str | os.PathLike[str] | None = None) -> Path:
        """Return the profiles directory, creating it if needed."""
        path = Path(override) if override else _user_config_base() / "profiles"
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
            path = dev_models if dev_models.exists() else _user_config_base() / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def launcher_config(override: str | os.PathLike[str] | None = None) -> Path:
        """Return the launcher app list config path (TOML)."""
        if override:
            return Path(override)
        return _user_config_base() / "launcher.toml"

    @staticmethod
    def gesture_mlp_model() -> Path:
        """Return the gesture MLP ONNX model path."""
        return Paths.models() / "gesture_mlp.onnx"

    @staticmethod
    def gesture_tcn_model() -> Path:
        """Return the gesture TCN ONNX model path."""
        return Paths.models() / "gesture_tcn_v1.onnx"

    @staticmethod
    def hand_landmarker() -> Path:
        """Return the MediaPipe hand landmarker task file path."""
        return Paths.models() / "hand_landmarker.task"

    @staticmethod
    def face_landmarker() -> Path:
        """Return the MediaPipe face landmarker task file path."""
        return Paths.models() / "face_landmarker.task"

    @staticmethod
    def l2cs_model() -> Path:
        """Return the L2CS-Net ONNX model path."""
        return Paths.models() / "l2cs_net_gaze360.onnx"

    @staticmethod
    def blaze_face_model() -> Path:
        """Return the BlazeFace short-range MediaPipe model path."""
        return Paths.models() / "blaze_face_short_range.tflite"

    @staticmethod
    def gaze_profile(name: str) -> Path:
        """Return the legacy (v1) gaze calibration profile path for *name*.

        v1 layout is a flat file under ``profiles/``:
            ``<profiles>/<name>.gaze.npz``

        v1.0+ runtimes still read this path (backward-compat), but new
        calibrations are written under :meth:`gaze_profile_v2` per
        ADR-0009. Use :meth:`gaze_profile_resolve` to get whichever
        exists for a given user/monitor.
        """
        return Paths.profiles() / f"{name}.gaze.npz"

    @staticmethod
    def gaze_profile_dir(user_id: str = "default", monitor_id: str | None = None) -> Path:
        """Return the v2 profile directory for ``<user>/<monitor>/``.

        Per ADR-0009. When *monitor_id* is None, the per-user directory
        ``<profiles>/<user>/`` is returned (used by the migrator to host
        a default "primary-legacy" subdirectory for migrated v1 files).
        Creates parents on first access.
        """
        base = Paths.profiles() / user_id
        if monitor_id is not None:
            base = base / monitor_id
        base.mkdir(parents=True, exist_ok=True)
        return base

    @staticmethod
    def gaze_profile_v2(
        user_id: str = "default",
        monitor_id: str = "primary-legacy",
        version: int = 1,
    ) -> Path:
        """Return the v2 ``.npz`` profile path ``<profiles>/<user>/<monitor>/v{N}.npz``.

        Per ADR-0009. The ``.meta.json`` sidecar lives next to the
        ``.npz`` with the same stem (``v{N}.meta.json``).
        """
        return Paths.gaze_profile_dir(user_id, monitor_id) / f"v{int(version)}.npz"

    @staticmethod
    def gaze_profile_history(
        user_id: str = "default",
        monitor_id: str = "primary-legacy",
    ) -> list[Path]:
        """Return v{N}.npz files for a profile, sorted by N ascending.

        Empty list when the directory does not exist or holds no
        v2 profiles. Useful for the HUD ("v3 active, 2 older versions")
        and for the ``profile migrate`` CLI command.
        """
        d = Paths.profiles() / user_id / monitor_id
        if not d.is_dir():
            return []
        candidates: list[tuple[int, Path]] = []
        for p in d.glob("v*.npz"):
            try:
                n = int(p.stem.lstrip("v"))
            except ValueError:
                continue
            candidates.append((n, p))
        candidates.sort(key=lambda x: x[0])
        return [p for _, p in candidates]

    @staticmethod
    def gaze_profile_latest_pointer(
        user_id: str = "default",
        monitor_id: str = "primary-legacy",
    ) -> Path:
        """Return the ``latest.txt`` pointer path inside a v2 profile dir.

        ``latest.txt`` is a one-line file containing the active version
        stem (e.g. ``v2``). Windows-safe alternative to symlinks
        (ADR-0009). The file is not created automatically — callers
        write it via atomic ``.part`` rename like the npz/meta files.
        """
        return Paths.gaze_profile_dir(user_id, monitor_id) / "latest.txt"

    @staticmethod
    def runtime_config(override: str | os.PathLike[str] | None = None) -> Path:
        """Return the runtime persistence file (TOML) path."""
        if override:
            return Path(override)
        base = _user_config_base()
        base.mkdir(parents=True, exist_ok=True)
        return base / "runtime.toml"
