"""AccelerationCurve — non-linear velocity-to-gain mapping for pointer control.

Maps hand movement velocity to a gain factor using a smooth exponential
interpolation curve:

- **Slow movements** → gain < 1.0 (damped for precision work).
- **Medium movements** → gain ≈ 1.0 (natural 1:1 feel).
- **Fast movements** → gain > 1.0 (amplified for quick navigation).

The curve is continuous and monotonically increasing, preventing abrupt
jumps when crossing velocity thresholds.

Usage::

    curve = AccelerationCurve(v_low=0.05, v_high=1.2, gain_low=0.7, gain_high=1.8)

    # velocity is the Euclidean speed of the normalised landmark (units/frame).
    gain = curve.gain(velocity=0.03)
    sx_gained = int(sx_raw * gain)
"""

from __future__ import annotations

import math


class AccelerationCurve:
    """Smooth velocity-to-gain mapping.

    Args:
        v_low:     Velocity below which ``gain_low`` is applied (units/frame).
        v_high:    Velocity above which ``gain_high`` is applied.
        gain_low:  Gain multiplier at slow speed (< 1.0 for precision dampening).
        gain_high: Gain multiplier at high speed (> 1.0 for fast navigation).

    The gain between *v_low* and *v_high* is interpolated with a smooth
    sigmoid so there are no hard transitions.
    """

    def __init__(
        self,
        v_low: float = 0.05,
        v_high: float = 1.2,
        gain_low: float = 0.7,
        gain_high: float = 1.8,
    ) -> None:
        self._v_low = float(v_low)
        self._v_high = float(max(v_high, v_low + 1e-6))
        self._gain_low = float(gain_low)
        self._gain_high = float(gain_high)

    def gain(self, velocity: float) -> float:
        """Return the gain factor for the given *velocity*.

        Args:
            velocity: Euclidean speed of the normalised landmark (≥ 0).

        Returns:
            Gain multiplier ∈ [``gain_low``, ``gain_high``].
        """
        v = abs(velocity)
        if v <= self._v_low:
            return self._gain_low
        if v >= self._v_high:
            return self._gain_high

        # Normalise velocity to [0, 1] within the transition band.
        t = (v - self._v_low) / (self._v_high - self._v_low)

        # Smooth-step (cubic Hermite): 3t² − 2t³ → avoids linear kink.
        t_smooth = t * t * (3.0 - 2.0 * t)

        return self._gain_low + (self._gain_high - self._gain_low) * t_smooth

    # Convenience: apply gain directly to (dx, dy) displacements.
    def apply(self, dx: float, dy: float) -> tuple[float, float]:
        """Scale displacement ``(dx, dy)`` by the velocity-appropriate gain.

        Args:
            dx: X displacement (signed, any unit consistent with v thresholds).
            dy: Y displacement.

        Returns:
            ``(dx_gained, dy_gained)``
        """
        v = math.sqrt(dx * dx + dy * dy)
        g = self.gain(v)
        return dx * g, dy * g
