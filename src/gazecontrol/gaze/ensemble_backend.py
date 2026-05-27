"""EnsembleBackend — weighted blend of two gaze backends.

Combines a landmark-based predictor (typically :class:`EyetraxBackend`) and
an appearance-based predictor (typically :class:`L2CSBackend`).

Both backends are queried each frame; weights renormalise dynamically when
one of them returns ``None`` so the consumer always gets a valid prediction
(provided at least one backend produced one).

Fusion modes (v1.0, G3)
-----------------------
The ``mode`` argument controls how the two backends are combined when
both produce a valid prediction:

- ``"static"`` — legacy behaviour. Per-frame weights equal the
  configured base weights, normalised so they sum to one. Each backend's
  self-reported confidence is averaged into the output confidence but
  does not influence the spatial blend.
- ``"confidence"`` — per-frame weights are the base weights rescaled by
  each backend's per-frame confidence (``w_i = base_w_i * conf_i / Σ``).
  A backend that reports zero confidence is effectively dropped for
  that frame. Default in v1.0 because it lets the ensemble lean on
  whichever backend is currently more sure of itself (head pose, lighting,
  occlusion).
- ``"kalman"`` — variance-weighted least-squares. When both predictions
  carry an ``uncertainty_px`` (from a GP / kernel mapper, ADR-0009),
  the spatial blend uses ``w_i ∝ 1 / σ_i²`` and the output
  ``uncertainty_px`` is the fused posterior sigma. When at least one
  prediction lacks uncertainty, the implementation gracefully falls
  back to ``"confidence"`` for that frame.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

import numpy as np

from gazecontrol.gaze.backend import GazeBackend, GazePrediction

logger = logging.getLogger(__name__)

EnsembleMode = Literal["static", "confidence", "kalman"]


class EnsembleBackend:
    """Blend two gaze backends with configurable weights and fusion mode."""

    name = "ensemble"

    def __init__(
        self,
        primary: GazeBackend,
        secondary: GazeBackend,
        weight_primary: float = 0.3,
        weight_secondary: float = 0.7,
        *,
        mode: EnsembleMode = "confidence",
    ) -> None:
        if weight_primary < 0 or weight_secondary < 0:
            raise ValueError("Ensemble weights must be non-negative.")
        if weight_primary + weight_secondary <= 0:
            raise ValueError("Ensemble weights cannot both be zero.")
        if mode not in ("static", "confidence", "kalman"):
            raise ValueError(f"Unknown ensemble mode: {mode!r}")
        self._primary = primary
        self._secondary = secondary
        self._w1 = float(weight_primary)
        self._w2 = float(weight_secondary)
        self._mode: EnsembleMode = mode

    @property
    def mode(self) -> EnsembleMode:
        """Active fusion mode (``"static"``, ``"confidence"``, or ``"kalman"``)."""
        return self._mode

    def start(self) -> bool:
        """Start both wrapped backends; succeed if at least one starts."""
        ok1 = self._safe_start(self._primary)
        ok2 = self._safe_start(self._secondary)
        if not ok1 and not ok2:
            logger.error("EnsembleBackend: neither backend started.")
            return False
        if not ok1:
            logger.warning("EnsembleBackend: primary backend disabled.")
        if not ok2:
            logger.warning("EnsembleBackend: secondary backend disabled.")
        return True

    def stop(self) -> None:
        """Stop both backends, tolerating individual failures."""
        for backend in (self._primary, self._secondary):
            try:
                backend.stop()
            except (RuntimeError, OSError):
                logger.exception("EnsembleBackend: stop failed for %s.", backend.name)

    def is_calibrated(self) -> bool:
        """True when at least one wrapped backend reports calibrated."""
        return self._primary.is_calibrated() or self._secondary.is_calibrated()

    def predict(
        self,
        frame_bgr: np.ndarray[Any, Any],
        frame_rgb: np.ndarray[Any, Any],
        timestamp: float,
    ) -> GazePrediction | None:
        """Blend predictions from both backends; renormalise on missing samples."""
        p1 = self._safe_predict(self._primary, frame_bgr, frame_rgb, timestamp)
        p2 = self._safe_predict(self._secondary, frame_bgr, frame_rgb, timestamp)

        # Blink propagates from any backend that reports it.
        any_blink = (p1 is not None and p1.blink) or (p2 is not None and p2.blink)
        if any_blink:
            base = p1 if (p1 is not None and p1.blink) else p2
            return GazePrediction(
                screen_xy=(0, 0),
                confidence=0.0,
                blink=True,
                backend_name=self.name,
                yaw_pitch_deg=base.yaw_pitch_deg if base is not None else None,
                quality_flags=base.quality_flags if base is not None else 0,
            )

        valid_p1 = p1 if (p1 is not None and not p1.blink) else None
        valid_p2 = p2 if (p2 is not None and not p2.blink) else None

        if valid_p1 is None and valid_p2 is None:
            return None
        if valid_p1 is None:
            assert valid_p2 is not None
            return self._passthrough(valid_p2)
        if valid_p2 is None:
            return self._passthrough(valid_p1)

        # Both valid → blend per the active mode.
        return self._blend(valid_p1, valid_p2)

    # ------------------------------------------------------------------
    # Fusion modes
    # ------------------------------------------------------------------

    def _blend(self, p1: GazePrediction, p2: GazePrediction) -> GazePrediction:
        """Combine *p1* and *p2* via the active ``mode``.

        Returns a fresh :class:`GazePrediction` tagged with the ensemble's
        own ``backend_name``. v1.0 enrichment fields (``head_pose_rad``,
        ``face_bbox_norm``, ``face_id``, ``quality_flags``) are propagated
        from the more confident input (or OR'd in the case of
        ``quality_flags``) so downstream consumers do not lose them.
        """
        uncertainty: float | None
        if self._mode == "kalman" and self._can_kalman(p1, p2):
            a, b, fused_sigma = self._kalman_weights(p1, p2)
            uncertainty = fused_sigma
        elif self._mode == "static":
            total = self._w1 + self._w2
            a = self._w1 / total
            b = self._w2 / total
            uncertainty = self._propagate_uncertainty(p1, p2, a, b)
        else:
            # "confidence" — also the kalman fallback path.
            a, b = self._confidence_weights(p1, p2)
            uncertainty = self._propagate_uncertainty(p1, p2, a, b)

        x = round(a * p1.screen_xy[0] + b * p2.screen_xy[0])
        y = round(a * p1.screen_xy[1] + b * p2.screen_xy[1])
        confidence = a * p1.confidence + b * p2.confidence

        # Carry v1.0 enrichment from the more heavily weighted backend.
        dominant = p1 if a >= b else p2
        return GazePrediction(
            screen_xy=(x, y),
            confidence=confidence,
            yaw_pitch_deg=p2.yaw_pitch_deg or p1.yaw_pitch_deg,
            blink=False,
            backend_name=self.name,
            uncertainty_px=uncertainty,
            head_pose_rad=dominant.head_pose_rad,
            face_bbox_norm=dominant.face_bbox_norm,
            face_id=dominant.face_id,
            quality_flags=int(p1.quality_flags | p2.quality_flags),
        )

    def _confidence_weights(
        self,
        p1: GazePrediction,
        p2: GazePrediction,
    ) -> tuple[float, float]:
        """Return per-frame weights for the ``"confidence"`` mode.

        ``w_i = base_w_i * conf_i / Σ``. When both effective weights
        collapse to zero (both backends reported zero confidence) we
        revert to the static base weights so the consumer still gets a
        sample — keeping the pointer alive matters more than a
        mathematically pure fallback.
        """
        a_raw = self._w1 * max(0.0, p1.confidence)
        b_raw = self._w2 * max(0.0, p2.confidence)
        total = a_raw + b_raw
        if total <= 0.0:
            base = self._w1 + self._w2
            return self._w1 / base, self._w2 / base
        return a_raw / total, b_raw / total

    def _kalman_weights(
        self,
        p1: GazePrediction,
        p2: GazePrediction,
    ) -> tuple[float, float, float]:
        """Return variance-weighted weights + fused sigma.

        Caller guarantees both predictions carry a positive
        ``uncertainty_px`` (enforced via :meth:`_can_kalman`).
        """
        assert p1.uncertainty_px is not None
        assert p2.uncertainty_px is not None
        s1 = max(float(p1.uncertainty_px), 1e-6)
        s2 = max(float(p2.uncertainty_px), 1e-6)
        v1 = s1 * s1
        v2 = s2 * s2
        # w_i ∝ 1 / σ_i² ; fused variance = (1 / (1/σ1² + 1/σ2²)).
        inv_v1 = 1.0 / v1
        inv_v2 = 1.0 / v2
        total = inv_v1 + inv_v2
        a = inv_v1 / total
        b = inv_v2 / total
        fused_sigma = math.sqrt(1.0 / total)
        return a, b, fused_sigma

    @staticmethod
    def _can_kalman(p1: GazePrediction, p2: GazePrediction) -> bool:
        """True when both predictions carry a positive ``uncertainty_px``."""
        return (
            p1.uncertainty_px is not None
            and p2.uncertainty_px is not None
            and p1.uncertainty_px > 0
            and p2.uncertainty_px > 0
        )

    @staticmethod
    def _propagate_uncertainty(
        p1: GazePrediction,
        p2: GazePrediction,
        a: float,
        b: float,
    ) -> float | None:
        """Best-effort fused sigma for non-Kalman blends.

        Treats the two predictions as independent Gaussians; returns
        ``sqrt(a²·σ1² + b²·σ2²)`` when both have a sigma, or whichever
        is available, or ``None`` when neither does.
        """
        s1 = p1.uncertainty_px
        s2 = p2.uncertainty_px
        if s1 is None and s2 is None:
            return None
        if s1 is None:
            assert s2 is not None
            return float(b * s2)
        if s2 is None:
            return float(a * s1)
        return float(math.sqrt(a * a * s1 * s1 + b * b * s2 * s2))

    def _passthrough(self, p: GazePrediction) -> GazePrediction:
        """Rewrap a single-backend prediction under the ensemble's name."""
        return GazePrediction(
            screen_xy=p.screen_xy,
            confidence=p.confidence,
            yaw_pitch_deg=p.yaw_pitch_deg,
            blink=False,
            backend_name=self.name,
            uncertainty_px=p.uncertainty_px,
            head_pose_rad=p.head_pose_rad,
            face_bbox_norm=p.face_bbox_norm,
            face_id=p.face_id,
            quality_flags=p.quality_flags,
        )

    # ------------------------------------------------------------------
    # Internals — start/predict failure tolerance
    # ------------------------------------------------------------------

    def _safe_start(self, backend: GazeBackend) -> bool:
        try:
            return bool(backend.start())
        except (RuntimeError, OSError, ValueError, AttributeError, ImportError):
            logger.exception("EnsembleBackend: start failed for %s.", backend.name)
            return False

    def _safe_predict(
        self,
        backend: GazeBackend,
        frame_bgr: np.ndarray[Any, Any],
        frame_rgb: np.ndarray[Any, Any],
        timestamp: float,
    ) -> GazePrediction | None:
        try:
            return backend.predict(frame_bgr, frame_rgb, timestamp)
        except (RuntimeError, ValueError):
            logger.debug("EnsembleBackend: predict failed for %s.", backend.name, exc_info=True)
            return None
