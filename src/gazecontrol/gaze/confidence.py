"""Per-frame confidence estimation for gaze backends (G1).

Replaces the v0.7/v0.8 hard-coded backend confidence with a transparent
blend of three signals:

1. **Face detector score** — directly from BlazeFace / Face Landmarker.
   Low when the detector is unsure (head turned, occlusion).
2. **Crop sharpness** — variance of the Laplacian of the cropped face.
   Low under motion blur, defocus, low light.
3. **Angle jitter** — stddev of the last N (yaw, pitch) samples.
   High when the gaze stream is flickering (eye lost, model noise).

The three signals are normalised into [0, 1] and combined into a
sigmoid so the output never saturates at exactly 0 or 1. Weights live
in :class:`ConfidenceModelSettings` so deployments can rebalance
without touching the backend.

Everything in this module is pure (no MediaPipe, no ONNX, no NumPy on
the hot path beyond ``np.var`` / Laplacian) so it can be unit-tested
under any environment. The Laplacian helper uses OpenCV when available
and falls back to a pure-NumPy implementation otherwise — useful in
test envs without ``cv2``.
"""

from __future__ import annotations

import collections
import math
from typing import Any

import numpy as np


def laplacian_variance(crop_bgr: np.ndarray[Any, Any]) -> float:
    """Return variance of the Laplacian of *crop_bgr*.

    A standard sharpness metric (Pertuz et al. 2013, "Analysis of focus
    measure operators for shape-from-focus"). Larger = sharper.

    Accepts H×W or H×W×3. Returns 0 for degenerate / empty inputs
    instead of raising — confidence should still be computable even if
    the crop is broken.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    arr = crop_bgr
    if arr.ndim == 3:
        # Convert to grayscale via luma weights. cv2.cvtColor would be
        # equivalent but we avoid a hard cv2 dep in this helper.
        arr = (0.114 * arr[..., 0] + 0.587 * arr[..., 1] + 0.299 * arr[..., 2]).astype(
            np.float32
        )
    try:
        import cv2

        lap = cv2.Laplacian(arr.astype(np.float32), cv2.CV_32F)
        return float(np.var(lap))
    except (ImportError, AttributeError):
        return float(_laplacian_variance_numpy(arr))


def _laplacian_variance_numpy(arr: np.ndarray[Any, Any]) -> float:
    """Fallback Laplacian implementation when cv2 is missing."""
    a = arr.astype(np.float32)
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    # 4-neighbour Laplacian: a[i-1,j] + a[i+1,j] + a[i,j-1] + a[i,j+1] − 4·a[i,j]
    lap = (
        a[:-2, 1:-1]
        + a[2:, 1:-1]
        + a[1:-1, :-2]
        + a[1:-1, 2:]
        - 4.0 * a[1:-1, 1:-1]
    )
    return float(lap.var())


class AngleJitter:
    """Rolling stddev of (yaw, pitch) samples for the jitter signal.

    The backend pushes one sample per frame; :meth:`score` returns a
    value in ``[0, 1]`` where 0 means "rock-solid" and 1 means "wildly
    jittery" (saturating at ``jitter_saturation_deg``).
    """

    def __init__(
        self,
        *,
        window: int = 5,
        jitter_saturation_deg: float = 8.0,
    ) -> None:
        self._buf: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=max(2, int(window))
        )
        self._sat = max(1e-6, float(jitter_saturation_deg))

    def reset(self) -> None:
        """Forget all samples — used by backends on stop/restart."""
        self._buf.clear()

    def push(self, yaw: float, pitch: float) -> None:
        """Add one (yaw, pitch) sample to the rolling buffer."""
        self._buf.append((float(yaw), float(pitch)))

    def score(self) -> float:
        """Return the current jitter score in ``[0, 1]``.

        Returns 0 before the buffer holds at least two samples (no
        information yet — should not penalise the first prediction).
        """
        if len(self._buf) < 2:
            return 0.0
        yaws = [p[0] for p in self._buf]
        pitches = [p[1] for p in self._buf]
        # Hypot of the two axis std-devs in degrees — symmetric in yaw/pitch.
        sig = math.hypot(_pop_stddev(yaws), _pop_stddev(pitches))
        return max(0.0, min(1.0, sig / self._sat))


def confidence_score(
    *,
    face_score: float,
    sharpness: float,
    jitter: float,
    sharpness_floor: float = 50.0,
    sharpness_ceiling: float = 500.0,
    w_face: float = 0.5,
    w_sharpness: float = 0.3,
    w_jitter: float = 0.2,
    bias: float = -0.5,
    floor: float = 0.05,
    ceiling: float = 0.95,
) -> float:
    """Combine the three signals into a confidence in ``[floor, ceiling]``.

    Signal preprocessing:

    * ``face_score`` is clamped to ``[0, 1]``.
    * ``sharpness`` is normalised by linear interpolation between
      ``sharpness_floor`` (→ 0) and ``sharpness_ceiling`` (→ 1).
    * ``jitter`` is clamped to ``[0, 1]`` and inverted (more jitter
      → lower confidence).

    The three normalised signals are weighted by ``w_face`` /
    ``w_sharpness`` / ``w_jitter`` and combined via a logistic
    sigmoid offset by ``bias``. The output is clipped to
    ``[floor, ceiling]`` so a single broken signal cannot drive the
    confidence to exactly 0 or 1 (the consumer's confidence-weighted
    fusion expects nonzero weights to keep predictions alive — see
    EnsembleBackend "confidence" mode, G3).
    """
    fs = max(0.0, min(1.0, float(face_score)))
    sharp_norm = _linear_clamp(
        float(sharpness),
        lo=float(sharpness_floor),
        hi=float(sharpness_ceiling),
    )
    jit_norm = 1.0 - max(0.0, min(1.0, float(jitter)))
    raw = w_face * fs + w_sharpness * sharp_norm + w_jitter * jit_norm + bias
    sig = _sigmoid(raw * 4.0)  # 4.0 stretches the [0,1] sum across the sigmoid curve
    return max(float(floor), min(float(ceiling), sig))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _linear_clamp(value: float, *, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    if value <= lo:
        return 0.0
    if value >= hi:
        return 1.0
    return (value - lo) / (hi - lo)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _pop_stddev(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return math.sqrt(sum((v - mean) ** 2 for v in xs) / n)
