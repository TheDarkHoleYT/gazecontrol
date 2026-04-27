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
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class GazePrediction:
    """Single gaze sample produced by a :class:`GazeBackend`.

    Attributes:
        screen_xy:        Predicted gaze point in screen pixels (clamped).
        confidence:       Backend self-reported confidence in ``[0, 1]``.
        yaw_pitch_deg:    Optional raw (yaw, pitch) angles in degrees, when available.
        blink:            True when the backend classified the frame as a blink.
        backend_name:     Name of the producing backend (for diagnostics/HUD).
    """

    screen_xy: tuple[int, int]
    confidence: float
    yaw_pitch_deg: tuple[float, float] | None = None
    blink: bool = False
    backend_name: str = ""


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
