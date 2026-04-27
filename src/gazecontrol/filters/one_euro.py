"""One Euro Filter — adaptive low-pass filter for real-time pointer smoothing.

Algorithm: Casiez et al. (2012) "1€ Filter: A Simple Speed-based Low-pass
Filter for Noisy Input in Interactive Systems".

Key insight:
- During slow/idle movement (low velocity): low cutoff → smooth, low jitter.
- During fast movement (high velocity): high cutoff → responsive, no lag.

Typical usage — one instance per axis::

    fx = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    fy = OneEuroFilter(min_cutoff=1.0, beta=0.007)

    for ts, raw_x, raw_y in stream:
        x = fx.filter(raw_x, timestamp=ts)
        y = fy.filter(raw_y, timestamp=ts)

Parameters (tuning guide)::

    min_cutoff (Hz): lower = smoother at rest, more lag on start.
                     Range 0.5–5.0; default 1.0.
    beta:            speed coefficient. Higher = more responsive to fast moves.
                     Range 0.0–0.5; default 0.007.
    d_cutoff (Hz):   cutoff for the derivative filter. Rarely needs tuning.
                     Default 1.0.
"""

from __future__ import annotations

import math
import time


class _LowPassFilter:
    """Single-pole IIR low-pass filter with adaptive frequency."""

    __slots__ = ("_cutoff", "_dx", "_freq", "_x")

    def __init__(self, cutoff: float, freq: float) -> None:
        self._freq = freq
        self._cutoff = cutoff
        self._x: float | None = None
        self._dx: float = 0.0

    def alpha(self, cutoff: float) -> float:
        """Compute smoothing factor α for the given cutoff frequency."""
        te = 1.0 / self._freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def step(self, x: float, cutoff: float) -> float:
        """Apply one filter step and return the smoothed value."""
        a = self.alpha(cutoff)
        if self._x is None:
            self._x = x
        self._x = a * x + (1.0 - a) * self._x
        return self._x

    @property
    def last(self) -> float:
        """Last filtered value (0.0 if filter was never stepped)."""
        return self._x if self._x is not None else 0.0


class OneEuroFilter:
    """Adaptive 1€ low-pass filter for a single scalar signal.

    Thread-unsafe — create one instance per thread or protect externally.

    Args:
        min_cutoff: Minimum cutoff frequency in Hz (lower = smoother at rest).
        beta:       Speed coefficient (higher = more responsive during motion).
        d_cutoff:   Cutoff for the derivative filter. Default 1.0.
        freq:       Initial sample rate estimate in Hz. Auto-updated from timestamps.

    Example::

        f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        smooth = f.filter(raw_value, timestamp=time.monotonic())
    """

    __slots__ = (
        "_beta",
        "_d_cutoff",
        "_dx_filter",
        "_freq",
        "_last_ts",
        "_min_cutoff",
        "_x_filter",
    )

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        freq: float = 30.0,
    ) -> None:
        self._freq = freq
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filter = _LowPassFilter(min_cutoff, freq)
        self._dx_filter = _LowPassFilter(d_cutoff, freq)
        self._last_ts: float | None = None

    def filter(self, x: float, timestamp: float | None = None) -> float:
        """Apply the filter to value *x* and return the smoothed result.

        Args:
            x:         Raw input value.
            timestamp: Monotonic timestamp in seconds.  ``time.monotonic()``
                       is used when *None*.

        Returns:
            Smoothed value.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        if self._last_ts is not None:
            dt = timestamp - self._last_ts
            if dt > 0.0:
                # Update frequency estimate from actual inter-frame interval.
                self._freq = 1.0 / dt
                self._x_filter._freq = self._freq
                self._dx_filter._freq = self._freq

        self._last_ts = timestamp

        # Estimate instantaneous derivative.
        prev_x = self._x_filter.last
        dx = (x - prev_x) * self._freq

        # Filter the derivative.
        edx = self._dx_filter.step(dx, self._d_cutoff)

        # Adaptive cutoff: rises with speed → reduces lag during fast motion.
        cutoff = self._min_cutoff + self._beta * abs(edx)

        return self._x_filter.step(x, cutoff)

    def reset(self) -> None:
        """Reset internal state (call when tracking is lost, then regained)."""
        self._x_filter._x = None
        self._dx_filter._x = None
        self._last_ts = None
