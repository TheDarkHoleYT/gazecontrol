"""EyetraxBackend — landmark-based gaze estimation via the eyetrax package.

eyetrax exposes a TinyMLP regressor that maps MediaPipe Face Mesh landmarks
to screen coordinates. It is light (CPU-only, ~5 ms / frame) and works
without GPU. Calibration profiles are stored as ``.pkl`` files under
``Paths.profiles()``.

This backend is optional: when the ``eyetrax`` package is missing, the
backend reports ``start() -> False`` and the pipeline falls back to other
gaze providers (or hand-only mode).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from gazecontrol.gaze.backend import GazePrediction
from gazecontrol.paths import Paths

logger = logging.getLogger(__name__)


class EyetraxBackend:
    """Gaze backend that wraps :mod:`eyetrax`'s ``GazeEstimator``."""

    name = "eyetrax"

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        profile_name: str = "default",
    ) -> None:
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._profile_name = profile_name
        self._estimator: Any = None
        self._calibrated = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Instantiate the eyetrax estimator and load the calibration profile."""
        try:
            from eyetrax import GazeEstimator
        except ImportError:
            logger.warning("eyetrax not installed — install with: pip install gazecontrol[eye]")
            return False

        try:
            self._estimator = GazeEstimator(
                model_name="tiny_mlp",
                model_kwargs={
                    "hidden_layer_sizes": (256, 128, 64),
                    "max_iter": 1000,
                    "alpha": 1e-4,
                    "early_stopping": True,
                },
            )
        except (RuntimeError, OSError, ValueError):
            logger.exception("EyetraxBackend: failed to instantiate GazeEstimator.")
            return False

        # Profile is optional at start-up — backend can run uncalibrated and
        # callers can decide whether to require calibration.
        profile = Paths.profiles() / f"{self._profile_name}.pkl"
        if profile.exists():
            try:
                self._estimator.load_model(str(profile))
                self._calibrated = True
                logger.info("EyetraxBackend: profile %s loaded.", profile)
            except (OSError, RuntimeError, ValueError):
                logger.exception("EyetraxBackend: profile load failed.")
                self._calibrated = False
        else:
            logger.warning("EyetraxBackend: profile %s not found; run --calibrate-gaze.", profile)
        return True

    def stop(self) -> None:
        """Release the estimator (no-op for eyetrax — let GC reclaim it)."""
        self._estimator = None
        self._calibrated = False

    def is_calibrated(self) -> bool:
        """True once a valid eyetrax profile has been loaded."""
        return self._calibrated

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        frame_bgr: np.ndarray[Any, Any],
        frame_rgb: np.ndarray[Any, Any],
        timestamp: float,
    ) -> GazePrediction | None:
        """Run eyetrax landmark extraction + regression on *frame_bgr*."""
        if self._estimator is None or not self._calibrated:
            return None
        try:
            features, blink = self._estimator.extract_features(frame_bgr)
        except (RuntimeError, ValueError):
            logger.debug("EyetraxBackend: extract_features failed.", exc_info=True)
            return None

        if blink:
            # Backend reports a blink; pipeline upstream applies blink-hold.
            return GazePrediction(
                screen_xy=(0, 0),
                confidence=0.0,
                blink=True,
                backend_name=self.name,
            )
        if features is None:
            return None
        try:
            px, py = self._estimator.predict([features])[0]
        except (RuntimeError, ValueError):
            logger.debug("EyetraxBackend: predict failed.", exc_info=True)
            return None
        x = max(0, min(self._screen_w - 1, int(px)))
        y = max(0, min(self._screen_h - 1, int(py)))
        # eyetrax does not surface a per-frame confidence; use 0.7 as a
        # constant prior for "valid landmarks present, no blink".
        return GazePrediction(
            screen_xy=(x, y),
            confidence=0.7,
            blink=False,
            backend_name=self.name,
        )
