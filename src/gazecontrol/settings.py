"""GazeControl settings — pydantic-settings based, env-var driven.

All settings can be overridden via environment variables with the prefix
``GAZECONTROL_``, using double underscore for nested groups::

    GAZECONTROL_CAMERA__INDEX=1
    GAZECONTROL_GAZE__MODEL_MODE=ensemble

An optional ``settings.toml`` in the project root (or current working
directory) is also loaded when present. See ``settings.toml.example``.
"""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class CameraSettings(BaseSettings):
    """Webcam capture parameters."""

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_CAMERA__", extra="ignore")

    index: int = Field(default=0, ge=0, description="OpenCV camera index")
    width: int = Field(default=1280, ge=320, le=3840)
    height: int = Field(default=720, ge=240, le=2160)
    fps: int = Field(default=30, ge=1, le=120)
    warmup_frames: int = Field(default=5, ge=0, le=60)


class GazeSettings(BaseSettings):
    """Gaze estimation and filtering parameters."""

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_GAZE__", extra="ignore")

    model_mode: Literal["mlp", "l2cs", "ensemble"] = Field(
        default="mlp",
        description="Active gaze model: mlp | l2cs | ensemble",
    )
    strict_l2cs: bool = Field(
        default=False,
        description="Raise ModelLoadError on L2CS load failure (vs. warn + disable)",
    )
    ensemble_weight_mlp: float = Field(default=0.7, ge=0.0, le=1.0)
    ensemble_weight_l2cs: float = Field(default=0.3, ge=0.0, le=1.0)

    one_euro_min_cutoff: float = Field(default=1.5, gt=0.0)
    one_euro_beta: float = Field(default=0.007, ge=0.0)
    one_euro_d_cutoff: float = Field(default=1.0, gt=0.0)

    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    @field_validator("ensemble_weight_mlp", "ensemble_weight_l2cs", mode="before")
    @classmethod
    def _clamp_weight(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @model_validator(mode="after")
    def _weights_sum(self) -> "GazeSettings":
        total = self.ensemble_weight_mlp + self.ensemble_weight_l2cs
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                "Ensemble weights sum to %.3f (expected 1.0) — normalizing.", total
            )
            self.ensemble_weight_mlp /= total
            self.ensemble_weight_l2cs /= total
        return self


class FixationSettings(BaseSettings):
    """I-VT fixation detector thresholds."""

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_FIXATION__", extra="ignore")

    velocity_threshold_deg_s: float = Field(default=30.0, gt=0.0)
    saccade_threshold_deg_s: float = Field(default=100.0, gt=0.0)
    min_fixation_duration_s: float = Field(default=0.1, ge=0.0)
    px_per_deg: float = Field(default=44.0, gt=0.0)


class DriftSettings(BaseSettings):
    """Drift correction parameters."""

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_DRIFT__", extra="ignore")

    edge_snap_margin_px: int = Field(default=80, ge=0)
    implicit_recal_weight: float = Field(default=0.15, ge=0.0, le=1.0)


class GestureSettings(BaseSettings):
    """Hand gesture detection and classification parameters."""

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_GESTURE__", extra="ignore")

    max_hands: int = Field(default=1, ge=1, le=2)
    min_detection_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    pinch_release_frames: int = Field(default=3, ge=1, le=30)

    drag_sensitivity: float = Field(default=1.5, gt=0.0)
    resize_sensitivity: float = Field(default=2.0, gt=0.0)
    swipe_velocity_threshold: float = Field(default=200.0, gt=0.0)


class IntentSettings(BaseSettings):
    """Intent state machine parameters."""

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_INTENT__", extra="ignore")

    dwell_time_s: float = Field(default=0.4, ge=0.0)
    ready_timeout_s: float = Field(default=3.0, ge=0.0)
    cooldown_s: float = Field(default=0.3, ge=0.0)
    blink_hold_max_s: float = Field(default=0.4, ge=0.0)
    double_pinch_window_s: float = Field(default=0.4, ge=0.0)


class OverlaySettings(BaseSettings):
    """HUD overlay visual parameters."""

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_OVERLAY__", extra="ignore")

    enabled: bool = True
    opacity: float = Field(default=0.85, ge=0.0, le=1.0)
    fps: int = Field(default=30, ge=1, le=60)

    gaze_dot_radius: int = Field(default=8, ge=1)
    gaze_dot_color: tuple[int, int, int] = (0, 220, 0)
    targeting_color: tuple[int, int, int] = (0, 100, 255)
    ready_color: tuple[int, int, int] = (255, 140, 0)


class AppSettings(BaseSettings):
    """Top-level application settings.

    Composes all sub-settings groups. Loaded from env vars and an optional
    ``settings.toml`` file in the working directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="GAZECONTROL_",
        env_nested_delimiter="__",
        toml_file="settings.toml",
        extra="ignore",
    )

    camera: CameraSettings = Field(default_factory=CameraSettings)
    gaze: GazeSettings = Field(default_factory=GazeSettings)
    fixation: FixationSettings = Field(default_factory=FixationSettings)
    drift: DriftSettings = Field(default_factory=DriftSettings)
    gesture: GestureSettings = Field(default_factory=GestureSettings)
    intent: IntentSettings = Field(default_factory=IntentSettings)
    overlay: OverlaySettings = Field(default_factory=OverlaySettings)

    log_level: str = Field(default="INFO")

    # Gesture label list — kept at app level for cross-module use
    gesture_labels: list[str] = Field(
        default=[
            "GRAB",
            "RELEASE",
            "PINCH",
            "SWIPE_LEFT",
            "SWIPE_RIGHT",
            "CLOSE_SIGN",
            "SCROLL_UP",
            "SCROLL_DOWN",
            "MAXIMIZE",
        ]
    )


# ---------------------------------------------------------------------------
# Module-level singleton — safe to import; instantiates lazily.
# ---------------------------------------------------------------------------
_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Return the process-wide settings singleton.

    Thread-safe for read access; call once at startup to initialise.
    """
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


def reset_settings(new: AppSettings | None = None) -> None:
    """Replace the settings singleton (test helper)."""
    global _settings
    _settings = new
