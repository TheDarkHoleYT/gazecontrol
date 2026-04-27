"""EnsembleBackend — weighted blend of two gaze backends.

Combines a landmark-based predictor (typically :class:`EyetraxBackend`) and
an appearance-based predictor (typically :class:`L2CSBackend`).

Both backends are queried each frame; weights renormalise dynamically when
one of them returns ``None`` so the consumer always gets a valid prediction
(provided at least one backend produced one).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from gazecontrol.gaze.backend import GazeBackend, GazePrediction

logger = logging.getLogger(__name__)


class EnsembleBackend:
    """Blend two gaze backends with configurable weights."""

    name = "ensemble"

    def __init__(
        self,
        primary: GazeBackend,
        secondary: GazeBackend,
        weight_primary: float = 0.3,
        weight_secondary: float = 0.7,
    ) -> None:
        if weight_primary < 0 or weight_secondary < 0:
            raise ValueError("Ensemble weights must be non-negative.")
        if weight_primary + weight_secondary <= 0:
            raise ValueError("Ensemble weights cannot both be zero.")
        self._primary = primary
        self._secondary = secondary
        self._w1 = float(weight_primary)
        self._w2 = float(weight_secondary)

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
            )

        valid_p1 = p1 if (p1 is not None and not p1.blink) else None
        valid_p2 = p2 if (p2 is not None and not p2.blink) else None

        if valid_p1 is None and valid_p2 is None:
            return None
        if valid_p1 is None:
            assert valid_p2 is not None
            return GazePrediction(
                screen_xy=valid_p2.screen_xy,
                confidence=valid_p2.confidence,
                yaw_pitch_deg=valid_p2.yaw_pitch_deg,
                blink=False,
                backend_name=self.name,
            )
        if valid_p2 is None:
            return GazePrediction(
                screen_xy=valid_p1.screen_xy,
                confidence=valid_p1.confidence,
                yaw_pitch_deg=valid_p1.yaw_pitch_deg,
                blink=False,
                backend_name=self.name,
            )

        # Both valid → weighted blend.
        total = self._w1 + self._w2
        a = self._w1 / total
        b = self._w2 / total
        x = round(a * valid_p1.screen_xy[0] + b * valid_p2.screen_xy[0])
        y = round(a * valid_p1.screen_xy[1] + b * valid_p2.screen_xy[1])
        confidence = a * valid_p1.confidence + b * valid_p2.confidence
        return GazePrediction(
            screen_xy=(x, y),
            confidence=confidence,
            yaw_pitch_deg=valid_p2.yaw_pitch_deg or valid_p1.yaw_pitch_deg,
            blink=False,
            backend_name=self.name,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _safe_start(self, backend: GazeBackend) -> bool:
        try:
            return bool(backend.start())
        except (RuntimeError, OSError, ValueError):
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
