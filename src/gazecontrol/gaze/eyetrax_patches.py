"""Deprecated shim — eyetrax patches moved to gazecontrol.gaze.compat.eyetrax."""
import warnings

warnings.warn(
    "gazecontrol.gaze.eyetrax_patches is deprecated. "
    "Use gazecontrol.gaze.compat.eyetrax instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gazecontrol.gaze.compat.eyetrax import PatchError, apply_patches  # noqa: E402

__all__ = ["PatchError", "apply_patches"]
