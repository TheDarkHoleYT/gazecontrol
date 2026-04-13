"""Deprecated configuration shim.

.. deprecated::
    Import from :mod:`gazecontrol.settings` and :mod:`gazecontrol.paths`
    instead. This module will be removed in v0.4.0.
"""
from __future__ import annotations

import warnings

from gazecontrol.paths import Paths
from gazecontrol.settings import get_settings

_DEPRECATION_MSG = (
    "gazecontrol.config is deprecated and will be removed in v0.4.0. "
    "Use gazecontrol.settings.get_settings() and gazecontrol.paths.Paths instead."
)


def _warn() -> None:
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# Legacy constants — backed by the settings singleton.
# Accessing any of these triggers a DeprecationWarning at runtime.
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> object:
    _warn()
    s = get_settings()
    _map: dict[str, object] = {
        # Camera
        "CAMERA_INDEX": s.camera.index,
        "FRAME_WIDTH": s.camera.width,
        "FRAME_HEIGHT": s.camera.height,
        "CAMERA_FPS": s.camera.fps,
        # Gaze
        "GAZE_MODEL_MODE": s.gaze.model_mode,
        "GAZE_1EURO_MIN_CUTOFF": s.gaze.one_euro_min_cutoff,
        "GAZE_1EURO_BETA": s.gaze.one_euro_beta,
        "GAZE_CONFIDENCE_THRESHOLD": s.gaze.confidence_threshold,
        "GAZE_ENSEMBLE_LANDMARK_WEIGHT": s.gaze.ensemble_weight_mlp,
        "GAZE_ENSEMBLE_APPEARANCE_WEIGHT": s.gaze.ensemble_weight_l2cs,
        # Paths
        "MODELS_DIR": str(Paths.models()),
        "PROFILES_DIR": str(Paths.profiles()),
        "L2CS_MODEL_PATH": str(Paths.l2cs_model()),
        "MLP_MODEL_PATH": str(Paths.gesture_mlp_model()),
        "LOG_FILE": str(Paths.log_file()),
        # Hand tracking
        "HAND_MAX_HANDS": s.gesture.max_hands,
        "HAND_MIN_DETECTION_CONFIDENCE": s.gesture.min_detection_confidence,
        "HAND_MIN_TRACKING_CONFIDENCE": s.gesture.min_tracking_confidence,
        "FEATURE_REF_WIDTH": s.camera.width,
        "FEATURE_REF_HEIGHT": s.camera.height,
        "SWIPE_VELOCITY_THRESHOLD": s.gesture.swipe_velocity_threshold,
        "DRAG_HAND_SENSITIVITY": s.gesture.drag_sensitivity,
        "RESIZE_HAND_SENSITIVITY": s.gesture.resize_sensitivity,
        # Gesture classifier
        "GESTURE_CONFIDENCE_THRESHOLD": s.gesture.confidence_threshold,
        "GESTURE_LABELS": s.gesture_labels,
        # Intent engine
        "DWELL_TIME_MS": int(s.intent.dwell_time_s * 1000),
        "READY_TIMEOUT_S": s.intent.ready_timeout_s,
        "COOLDOWN_MS": int(s.intent.cooldown_s * 1000),
        # Overlay
        "OVERLAY_GAZE_DOT_RADIUS": s.overlay.gaze_dot_radius,
        "OVERLAY_GAZE_DOT_COLOR": s.overlay.gaze_dot_color,
        "OVERLAY_TARGETING_COLOR": s.overlay.targeting_color,
        "OVERLAY_READY_COLOR": s.overlay.ready_color,
        # Logging
        "LOG_LEVEL": s.log_level,
    }
    if name in _map:
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
