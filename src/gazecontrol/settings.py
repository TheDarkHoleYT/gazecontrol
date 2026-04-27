"""GazeControl settings — pydantic-settings based, env-var driven.

All settings can be overridden via environment variables with the prefix
``GAZECONTROL_``, using double underscore for nested groups::

    GAZECONTROL_CAMERA__INDEX=1
    GAZECONTROL_INTERACTION__TAP_MS=180
    GAZECONTROL_RUNTIME__INPUT_MODE=eye-hand

An optional ``settings.toml`` in the project root (or current working
directory) is also loaded when present. See ``settings.toml.example``.
"""

from __future__ import annotations

import threading
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class InputMode(StrEnum):
    """Selectable input modes for the GazeControl pipeline."""

    HAND_ONLY = "hand"
    EYE_HAND = "eye-hand"


class LoggingSettings(BaseModel):
    """Logging configuration (format, rotation, verbosity)."""

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    format: str = Field(default="text", pattern="^(text|json)$")
    rotation_mb: int = Field(default=5, ge=1, le=100)
    backup_count: int = Field(default=5, ge=0, le=50)
    profiler_log_every_n: int = Field(
        default=300,
        ge=1,
        description="Log pipeline timing every N ticks (approx every 10 s at 30 fps).",
    )


class CameraSettings(BaseSettings):
    """Webcam capture parameters."""

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_CAMERA__", extra="ignore")

    index: int = Field(default=0, ge=0, description="OpenCV camera index")
    width: int = Field(default=1280, ge=320, le=3840)
    height: int = Field(default=720, ge=240, le=2160)
    fps: int = Field(default=30, ge=1, le=120)
    warmup_frames: int = Field(
        default=20,
        ge=0,
        le=60,
        description=(
            "Frames to drain after open() before trusting pixel data. "
            "Windows/DSHOW cameras return black frames for ~10-20 reads "
            "after cap.set() calls — increase if preview starts black."
        ),
    )
    auto_exposure: str = Field(
        default="auto",
        description=(
            "auto: let the OS manage exposure (recommended). "
            "manual: force DSHOW manual-exposure flag (legacy behaviour)."
        ),
    )
    min_brightness_ok: float = Field(
        default=15.0,
        ge=0.0,
        le=255.0,
        description="Minimum mean-pixel brightness for a frame to be considered usable.",
    )
    blur_threshold: float = Field(
        default=15.0,
        ge=0.0,
        description=(
            "Minimum Laplacian variance for a frame to be considered sharp enough. "
            "Typical USB webcam frames score 15–40; set to 0.0 to disable the check."
        ),
    )
    enhance: bool = Field(
        default=False,
        description=(
            "Apply CLAHE + sharpening to each frame before hand detection. "
            "Helps in poor lighting but costs ~3–5 ms/frame. "
            "Disabled by default — MediaPipe handles variable lighting well."
        ),
    )


class OneEuroSettings(BaseModel):
    """Parameters for the 1€ adaptive low-pass filter (Casiez et al. 2012).

    Lower ``min_cutoff`` → smoother at rest, higher ``beta`` → more responsive
    during fast motion.  Rarely needs tuning beyond these two values.
    """

    enabled: bool = True
    min_cutoff: float = Field(
        default=1.0,
        gt=0.0,
        description="Minimum cutoff frequency (Hz). Lower = smoother at rest.",
    )
    beta: float = Field(
        default=0.007,
        ge=0.0,
        description="Speed coefficient. Higher = faster response during motion.",
    )
    d_cutoff: float = Field(
        default=1.0,
        gt=0.0,
        description="Cutoff for the derivative filter. Rarely needs tuning.",
    )


class KalmanPointerSettings(BaseModel):
    """Parameters for the constant-velocity Kalman filter."""

    enabled: bool = True
    process_noise: float = Field(
        default=1e-3,
        gt=0.0,
        description=(
            "Process noise variance Q. Lower = smoother prediction, "
            "slower to adapt to direction changes."
        ),
    )
    measurement_noise_base: float = Field(
        default=1.0,
        gt=0.0,
        description=("Base measurement noise variance R. Scaled by 1/confidence at each update."),
    )


class AccelerationCurveSettings(BaseModel):
    """Non-linear velocity-to-gain mapping for pointer acceleration."""

    enabled: bool = True
    v_low: float = Field(
        default=0.05,
        ge=0.0,
        description="Velocity below which gain_low is applied (normalised units/frame).",
    )
    v_high: float = Field(
        default=1.2,
        gt=0.0,
        description="Velocity above which gain_high is applied.",
    )
    gain_low: float = Field(
        default=0.7,
        gt=0.0,
        description="Gain multiplier at slow speed (< 1.0 for precision dampening).",
    )
    gain_high: float = Field(
        default=1.8,
        gt=0.0,
        description="Gain multiplier at high speed (> 1.0 for fast navigation).",
    )


class DeadZoneSettings(BaseModel):
    """Circular dead-zone with hysteresis to suppress micro-jitter."""

    enabled: bool = True
    radius_px: float = Field(
        default=3.0,
        ge=0.0,
        description="Inner dead-zone radius in pixels. Movements smaller than this are suppressed.",
    )
    hysteresis_px: float = Field(
        default=6.0,
        ge=0.0,
        description=(
            "Outer radius — the lock is released only when movement exceeds "
            "this distance. Must be ≥ radius_px."
        ),
    )


class PointerFilterSettings(BaseModel):
    """Aggregated pointer filter stack.

    Applied in order: One-Euro → Kalman → dead-zone → acceleration curve.
    Each sub-filter can be individually enabled/disabled.
    Override via env vars using the ``GAZECONTROL_GESTURE__POINTER__`` prefix::

        GAZECONTROL_GESTURE__POINTER__ONE_EURO__BETA=0.02
        GAZECONTROL_GESTURE__POINTER__KALMAN__ENABLED=false
    """

    one_euro: OneEuroSettings = Field(default_factory=OneEuroSettings)
    kalman: KalmanPointerSettings = Field(default_factory=KalmanPointerSettings)
    acceleration: AccelerationCurveSettings = Field(default_factory=AccelerationCurveSettings)
    dead_zone: DeadZoneSettings = Field(default_factory=DeadZoneSettings)


class GestureSettings(BaseSettings):
    """Hand gesture detection and classification parameters."""

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_GESTURE__", extra="ignore")

    max_hands: int = Field(default=1, ge=1, le=2)
    min_detection_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    swipe_velocity_threshold: float = Field(default=200.0, gt=0.0)
    pointer_sensitivity: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description=(
            "Pointer movement amplification. "
            "1.0 = full camera frame maps to full screen. "
            "2.0 = move hand in central 50% of frame to cover full screen (default). "
            "3.0 = central 33%, etc. Increase if hand movement feels too large."
        ),
    )
    pointer: PointerFilterSettings = Field(
        default_factory=PointerFilterSettings,
        description="Pointer filter stack (One-Euro, Kalman, acceleration, dead-zone).",
    )


class InteractionSettings(BaseSettings):
    """Pinch-interaction timing and geometry parameters."""

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_INTERACTION__", extra="ignore")

    pinch_down_threshold: float = Field(
        default=0.22,
        ge=0.0,
        le=1.0,
        description="Normalized aperture below which a pinch is detected.",
    )
    pinch_up_threshold: float = Field(
        default=0.32,
        ge=0.0,
        le=1.0,
        description="Normalized aperture above which the pinch is released.",
    )
    tap_ms: int = Field(
        default=220,
        ge=50,
        le=600,
        description="Max milliseconds for a quick tap (vs a held pinch).",
    )
    hold_ms: int = Field(
        default=280,
        ge=100,
        le=1000,
        description="Milliseconds of continuous pinch before drag/resize begins.",
    )
    double_pinch_ms: int = Field(
        default=420,
        ge=100,
        le=1000,
        description="Max milliseconds between two taps to trigger a double-pinch.",
    )
    tap_max_move_px: int = Field(
        default=18,
        ge=0,
        description="Max pixel movement during a tap before it is discarded.",
    )
    grip_ratio: float = Field(
        default=0.18,
        ge=0.05,
        le=0.5,
        description="Fraction of each window dimension that is the resize grip zone.",
    )
    cooldown_ms: int = Field(
        default=120,
        ge=0,
        le=1000,
        description="Milliseconds in COOLDOWN before returning to IDLE.",
    )
    min_window_width: int = Field(
        default=200, ge=50, description="Minimum window width during resize."
    )
    min_window_height: int = Field(
        default=150, ge=50, description="Minimum window height during resize."
    )


class LauncherAppModel(BaseModel):
    """Configuration for a single launchable application."""

    name: str
    exe: str
    args: list[str] = Field(default_factory=list)
    icon: str | None = None


class LauncherSettings(BaseSettings):
    """Launcher overlay configuration."""

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_LAUNCHER__", extra="ignore")

    apps: list[LauncherAppModel] = Field(default_factory=list)
    columns: int = Field(default=4, ge=1, le=8)
    panel_opacity: float = Field(default=0.85, ge=0.0, le=1.0)


class OverlaySettings(BaseSettings):
    """HUD overlay visual parameters."""

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_OVERLAY__", extra="ignore")

    enabled: bool = True
    opacity: float = Field(default=0.85, ge=0.0, le=1.0)
    fps: int = Field(default=30, ge=1, le=60)

    pointer_radius: int = Field(default=8, ge=1, description="Radius of the fingertip cursor dot.")
    pointer_color: tuple[int, int, int] = (0, 220, 100)
    grip_hint_color: tuple[int, int, int] = (255, 160, 0)
    drag_color: tuple[int, int, int] = (0, 160, 255)
    resize_color: tuple[int, int, int] = (200, 80, 255)
    targeting_color: tuple[int, int, int] = (0, 100, 255)


class DriftCorrectorSettings(BaseModel):
    """Drift-correction parameters for gaze tracking.

    Edge snapping nudges the gaze offset when the predicted point
    overshoots the screen border. Implicit recalibration uses the
    centroid of windows the user actively interacts with as ground truth.
    """

    enabled: bool = True
    edge_margin_px: int = Field(default=60, ge=0)
    edge_correction_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    implicit_alpha: float = Field(default=0.08, gt=0.0, le=1.0)
    max_correction_px: int = Field(default=200, ge=0)


class GazeSettings(BaseSettings):
    """Eye tracker (gaze estimation) parameters.

    Active only when ``RuntimeSettings.input_mode == InputMode.EYE_HAND``.
    """

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_GAZE__", extra="ignore")

    backend: Literal["eyetrax", "l2cs", "ensemble"] = Field(
        default="ensemble",
        description="Which gaze backend to use; ensemble blends eyetrax + L2CS.",
    )
    ensemble_weight_eyetrax: float = Field(default=0.3, ge=0.0, le=1.0)
    ensemble_weight_l2cs: float = Field(default=0.7, ge=0.0, le=1.0)
    one_euro_min_cutoff: float = Field(default=1.5, gt=0.0)
    one_euro_beta: float = Field(default=0.007, ge=0.0)
    blink_hold_max_s: float = Field(
        default=0.4,
        gt=0.0,
        description="Hold last gaze sample for up to this many seconds during a blink.",
    )
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    fixation_velocity_deg_s: float = Field(default=30.0, gt=0.0)
    saccade_velocity_deg_s: float = Field(default=100.0, gt=0.0)
    px_per_deg: float = Field(
        default=44.0,
        gt=0.0,
        description="Pixels per visual degree (≈ 24'' FHD at 60 cm).",
    )
    drift: DriftCorrectorSettings = Field(default_factory=DriftCorrectorSettings)
    strict_l2cs: bool = Field(
        default=False,
        description="Raise on missing L2CS model instead of falling back to eyetrax.",
    )
    profile: str = Field(
        default="default",
        description="Calibration profile name (saved under <user_config>/profiles).",
    )


class FusionSettings(BaseSettings):
    """Pointer fusion parameters for ``EYE_HAND`` mode.

    Hand has priority over gaze for fine pointing; gaze provides hover and
    target-at-distance. ``gaze_assisted_click`` lets a stationary hand
    delegate click targets to the current gaze fixation.
    """

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_FUSION__", extra="ignore")

    hand_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    gaze_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    divergence_threshold_px: int = Field(default=300, ge=0)
    gaze_assisted_click: bool = Field(
        default=False,
        description="When hand is idle, use gaze fixation centroid for click target.",
    )


class RuntimeSettings(BaseSettings):
    """Runtime selection (input mode) and persisted user choice.

    The mode selector dialog updates ``last_chosen_mode`` and writes a
    small TOML file under the user-config directory. CLI ``--mode`` and
    the ``GAZECONTROL_RUNTIME__INPUT_MODE`` env var override the file.
    """

    model_config = SettingsConfigDict(env_prefix="GAZECONTROL_RUNTIME__", extra="ignore")

    input_mode: InputMode = Field(default=InputMode.HAND_ONLY)
    mode_selector_remember: bool = Field(default=True)
    last_chosen_mode: InputMode | None = Field(default=None)
    show_mode_selector: bool = Field(
        default=True,
        description="Show the mode-selector dialog at startup; disable for CI/scripting.",
    )


class GestureLabelsSettings(BaseModel):
    """Ordered list of gesture label names used by classifiers.

    Defaults to the full :data:`~gazecontrol.gesture.labels.DEFAULT_LABELS`
    vocabulary.  Override to restrict which labels the MLP model produces.
    """

    labels: list[str] = Field(
        default_factory=lambda: [
            "PINCH",
            "RELEASE",
            "SCROLL_UP",
            "SCROLL_DOWN",
            "SWIPE_LEFT",
            "SWIPE_RIGHT",
            "GRAB",
            "CLOSE_SIGN",
            "MAXIMIZE",
        ]
    )


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
    gesture: GestureSettings = Field(default_factory=GestureSettings)
    interaction: InteractionSettings = Field(default_factory=InteractionSettings)
    launcher: LauncherSettings = Field(default_factory=LauncherSettings)
    overlay: OverlaySettings = Field(default_factory=OverlaySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    gesture_labels: GestureLabelsSettings = Field(default_factory=GestureLabelsSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    gaze: GazeSettings = Field(default_factory=GazeSettings)
    fusion: FusionSettings = Field(default_factory=FusionSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Configure TOML + env-var settings sources."""
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )


# ---------------------------------------------------------------------------
# Module-level singleton — safe to import; instantiates lazily.
# ---------------------------------------------------------------------------
_settings: AppSettings | None = None
_settings_lock = threading.Lock()


def get_settings() -> AppSettings:
    """Return the process-wide settings singleton.

    Thread-safe: uses a double-checked lock so concurrent first calls do not
    each construct a separate AppSettings (which could trigger conflicting
    env-var reads / TOML parses).
    """
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = AppSettings()
    return _settings


def reset_settings(new: AppSettings | None = None) -> None:
    """Replace the settings singleton (test helper)."""
    global _settings
    _settings = new
