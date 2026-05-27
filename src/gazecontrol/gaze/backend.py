"""GazeBackend Protocol and shared value types.

Every concrete eye-tracking backend (eyetrax, L2CS-Net, ensemble of both)
implements this Protocol so the pipeline stage can swap them without
conditional code.

Threading model
---------------
``start()`` / ``stop()`` are called from the pipeline thread. ``predict()``
is called once per frame, also from the pipeline thread. Implementations
must allocate heavy resources (ONNX session, MediaPipe Face Mesh) inside
``start()`` to honour the legacy single-thread contract enforced by
MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Any, Protocol, runtime_checkable

import numpy as np


class GazeQuality(IntFlag):
    """Bitfield describing per-frame gaze prediction quality.

    Backends OR together the flags that apply to the current frame and
    expose the combined value via :attr:`GazePrediction.quality_flags`.
    The default value of ``0`` (no flags set) means "nominal — no
    quality concerns reported".
    """

    NONE = 0
    BLINK = 1 << 0
    LOW_LIGHT = 1 << 1
    OFF_AXIS = 1 << 2
    OCCLUDED = 1 << 3
    MULTI_FACE = 1 << 4


@dataclass(frozen=True)
class GazePrediction:
    """Single gaze sample produced by a :class:`GazeBackend`.

    Attributes:
        screen_xy:        Predicted gaze point in screen pixels (clamped).
        confidence:       Backend self-reported confidence in ``[0, 1]``.
        yaw_pitch_deg:    Optional raw (yaw, pitch) angles in degrees, when available.
        blink:            True when the backend classified the frame as a blink.
        backend_name:     Name of the producing backend (for diagnostics/HUD).
        uncertainty_px:   Predicted screen-space sigma in pixels (Gaussian-process
                          mappers populate this; classical mappers leave it None).
        head_pose_rad:    (yaw, pitch, roll) head pose in radians, computed from
                          face landmarks (typically via solvePnP). ``None`` when
                          the backend cannot estimate it for the current frame.
        face_bbox_norm:   (x_min, y_min, x_max, y_max) bounding box of the tracked
                          face in normalised [0, 1] frame coords. ``None`` when
                          no face was detected (e.g. landmark-only backends).
        face_id:          Stable id of the tracked face across frames (set by
                          backends with multi-face disambiguation). ``None`` when
                          tracking is not active.
        quality_flags:    OR-combination of :class:`GazeQuality` bits describing
                          known quality concerns for the current frame. Defaults
                          to ``0`` (no flags) for nominal samples.
    """

    screen_xy: tuple[int, int]
    confidence: float
    yaw_pitch_deg: tuple[float, float] | None = None
    blink: bool = False
    backend_name: str = ""
    uncertainty_px: float | None = None
    head_pose_rad: tuple[float, float, float] | None = None
    face_bbox_norm: tuple[float, float, float, float] | None = None
    face_id: int | None = None
    quality_flags: int = 0


@runtime_checkable
class GazeBackend(Protocol):
    """Interface for gaze-estimation backends."""

    @property
    def name(self) -> str:
        """Stable identifier (e.g. ``"eyetrax"``, ``"l2cs"``, ``"ensemble"``)."""
        ...

    def start(self) -> bool:
        """Allocate resources and load any persisted calibration profile.

        Returns:
            True on success. False signals an unrecoverable failure
            (caller should fall back to hand-only mode).
        """
        ...

    def stop(self) -> None:
        """Release resources. Must be idempotent."""
        ...

    def is_calibrated(self) -> bool:
        """True when a usable calibration profile is loaded."""
        ...

    def predict(
        self,
        frame_bgr: np.ndarray[Any, Any],
        frame_rgb: np.ndarray[Any, Any],
        timestamp: float,
    ) -> GazePrediction | None:
        """Estimate gaze for the current frame, or return ``None`` if unavailable."""
        ...
