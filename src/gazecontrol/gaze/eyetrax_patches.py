"""
Isolates the eyetrax monkey-patching required to make calibration work on Windows.

eyetrax uses its own camera-open logic (which often picks MSMF on Windows and hangs),
and its own face-detection waiting routine (which uses a pure-black background that
causes webcam auto-exposure to blow out).  We replace those two internal functions with
our improved versions.

Usage
-----
Call `apply_patches(open_camera_fn, wait_for_face_fn)` once at startup, before calling
any eyetrax calibration routine.  Both callables must be already defined; this module
does not define them.
"""
from __future__ import annotations

import importlib.metadata
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# eyetrax version range this patching has been tested against.
_TESTED_VERSIONS = {"0.4.0", "0.4.1", "0.5.0"}

_PATCH_TARGETS: list[tuple[str, str]] = [
    ("eyetrax.calibration.dense_grid", "open_camera"),
    ("eyetrax.calibration.adaptive", "open_camera"),
    ("eyetrax.calibration.common", "wait_for_face_and_countdown"),
]


class PatchError(RuntimeError):
    """Raised when a required eyetrax internal module cannot be patched."""


def apply_patches(
    open_camera_fn: Callable,
    wait_for_face_fn: Callable,
) -> None:
    """Monkey-patch eyetrax calibration internals.

    Parameters
    ----------
    open_camera_fn:
        Replacement for ``eyetrax.calibration.*.open_camera``.
        Signature: ``(index: int) -> cv2.VideoCapture``.
    wait_for_face_fn:
        Replacement for ``eyetrax.calibration.common.wait_for_face_and_countdown``.
        Signature: ``(cap, gaze_estimator, screen_w, screen_h, dur) -> bool``.

    Raises
    ------
    PatchError
        If any target module cannot be imported.  This is a hard failure: without the
        patches, calibration on Windows will likely hang or produce bad frames.
    """
    _check_version()

    replacements: dict[str, Callable] = {
        "open_camera": open_camera_fn,
        "wait_for_face_and_countdown": wait_for_face_fn,
    }

    for module_path, attr in _PATCH_TARGETS:
        try:
            module = _import_module(module_path)
        except ImportError as exc:
            raise PatchError(
                f"Cannot import eyetrax internal module '{module_path}'. "
                f"eyetrax may have been updated and moved this symbol. "
                f"Original error: {exc}"
            ) from exc
        setattr(module, attr, replacements[attr])
        logger.debug("Patched %s.%s", module_path, attr)

    logger.info(
        "eyetrax calibration patches applied (%d targets)", len(_PATCH_TARGETS)
    )


def _import_module(dotted_path: str):
    """Import a module by dotted path, raising ImportError if it does not exist."""
    import importlib
    return importlib.import_module(dotted_path)


def _check_version() -> None:
    """Emit a warning if the installed eyetrax version is not in the tested set."""
    try:
        version = importlib.metadata.version("eyetrax")
    except importlib.metadata.PackageNotFoundError:
        logger.warning("eyetrax package metadata not found; skipping version check")
        return

    if version not in _TESTED_VERSIONS:
        logger.warning(
            "eyetrax version %s has not been tested with these patches "
            "(tested: %s). Calibration may behave unexpectedly.",
            version,
            ", ".join(sorted(_TESTED_VERSIONS)),
        )
    else:
        logger.debug("eyetrax version %s is in the tested set", version)
