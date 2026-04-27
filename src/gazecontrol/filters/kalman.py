"""KalmanFilter2D — constant-velocity Kalman filter for 2-D screen coordinates.

State vector: [x, y, vx, vy]  (position + velocity in pixels/frame).

The filter predicts one step ahead each frame (compensating pipeline latency)
then updates with the measured landmark position.

Measurement noise *R* can be scaled per-frame by the MediaPipe hand-presence
confidence: low confidence → high noise → prediction trusted more.

Usage::

    kf = KalmanFilter2D(process_noise=1e-3, measurement_noise_base=1.0)

    # First call: filter initialises from the measurement.
    sx, sy = kf.update(raw_x, raw_y)

    # Subsequent calls (pass confidence ∈ [0, 1]).
    sx, sy = kf.update(raw_x, raw_y, confidence=0.95)

    # When tracking is lost, reset so the next detection starts clean.
    kf.reset()
"""

from __future__ import annotations

from typing import Any

import numpy as np


class KalmanFilter2D:
    """Constant-velocity Kalman filter for (x, y) screen coordinates.

    Args:
        process_noise:         Variance of the process noise Q (controls how
                               much the velocity is allowed to change each
                               frame).  Lower = smoother, slower to adapt.
        measurement_noise_base: Base variance of the measurement noise R.
                               Scaled by ``1 / confidence`` at update time so
                               that low-confidence detections are trusted less.

    State: ``[x, y, vx, vy]`` where velocity is in pixels per update.
    """

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise_base: float = 1.0,
    ) -> None:
        self._q = float(process_noise)
        self._r_base = float(measurement_noise_base)

        # State vector [x, y, vx, vy].
        self._x = np.zeros(4, dtype=np.float64)

        # State covariance (large initial uncertainty).
        self._P = np.eye(4, dtype=np.float64) * 1000.0

        # State-transition matrix: constant-velocity model.
        self._F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )

        # Measurement matrix: we observe (x, y) only.
        self._H = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float64,
        )

        # Process noise covariance Q.
        self._Q = np.eye(4, dtype=np.float64) * self._q

        # Measurement noise covariance R (scaled at update time).
        self._R_base = np.eye(2, dtype=np.float64) * self._r_base

        self._initialised = False

    # ------------------------------------------------------------------

    def update(
        self,
        x: float,
        y: float,
        confidence: float = 1.0,
    ) -> tuple[float, float]:
        """Predict + update the filter and return smoothed ``(x, y)``.

        Args:
            x:          Measured x coordinate (pixels).
            y:          Measured y coordinate (pixels).
            confidence: MediaPipe detection confidence ∈ [0, 1].
                        Low values inflate measurement noise so the predict
                        step is trusted more.

        Returns:
            ``(sx, sy)`` — filtered screen coordinates.
        """
        if not self._initialised:
            self._x[:] = [x, y, 0.0, 0.0]
            self._initialised = True
            return float(x), float(y)

        # --- Predict ---
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # --- Update ---
        z = np.array([x, y], dtype=np.float64)
        # Scale R by inverse confidence (clipped so it never explodes).
        scale = 1.0 / max(float(confidence), 0.05)
        R = self._R_base * scale

        S = self._H @ P_pred @ self._H.T + R
        K = P_pred @ self._H.T @ np.linalg.inv(S)
        innovation = z - self._H @ x_pred
        self._x = x_pred + K @ innovation
        I4: np.ndarray[Any, Any] = np.eye(4, dtype=np.float64)
        self._P = (I4 - K @ self._H) @ P_pred

        return float(self._x[0]), float(self._x[1])

    def predict_only(self) -> tuple[float, float]:
        """Run predict step without a measurement (for frames where hand is lost briefly).

        Returns:
            ``(px, py)`` — extrapolated position.
        """
        if not self._initialised:
            return 0.0, 0.0
        x_pred = self._F @ self._x
        self._x = x_pred
        self._P = self._F @ self._P @ self._F.T + self._Q
        return float(self._x[0]), float(self._x[1])

    def reset(self) -> None:
        """Reset filter state (call when the hand is lost for more than ~0.5 s)."""
        self._x[:] = 0.0
        self._P = np.eye(4, dtype=np.float64) * 1000.0
        self._initialised = False
