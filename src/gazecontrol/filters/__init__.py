"""gazecontrol.filters — pointer-smoothing and motion filters.

Provides adaptive real-time filters for fingertip coordinate smoothing:

- :class:`OneEuroFilter` — speed-based low-pass (Casiez et al. 2012).
  Smooth at rest, responsive during fast motion.
- :class:`KalmanFilter2D` — constant-velocity Kalman for 2-D screen coords.
  Predicts one frame ahead to compensate pipeline latency.
- :class:`AccelerationCurve` — non-linear velocity-to-gain mapping.
  Slow movements are precision-dampened; fast movements are amplified.
- :class:`DeadZone` — circular dead-zone with hysteresis.
  Suppresses micro-jitter when the hand is nearly stationary.
"""

from gazecontrol.filters.acceleration_curve import AccelerationCurve
from gazecontrol.filters.dead_zone import DeadZone
from gazecontrol.filters.kalman import KalmanFilter2D
from gazecontrol.filters.one_euro import OneEuroFilter

__all__ = [
    "AccelerationCurve",
    "DeadZone",
    "KalmanFilter2D",
    "OneEuroFilter",
]
