"""PinchTracker — converts continuous pinch-aperture readings to discrete events.

Applies a two-threshold hysteresis filter so that noisy frames near the
pinch boundary do not generate spurious ``DOWN``/``UP`` flickers.

    - Aperture < ``down_threshold`` while NOT pinching  → emit ``PinchEvent.DOWN``
    - Aperture > ``up_threshold``   while pinching      → emit ``PinchEvent.UP``
    - Otherwise                                         → emit ``PinchEvent.HOLD`` (if pinching)
                                                          or ``PinchEvent.NONE``  (if not pinching)
"""

from __future__ import annotations

import time
from enum import StrEnum


class PinchEvent(StrEnum):
    """Discrete pinch event types produced by :class:`PinchTracker`."""

    NONE = "NONE"  # hand visible but pinch not active
    DOWN = "DOWN"  # pinch just started
    HOLD = "HOLD"  # pinch in progress
    UP = "UP"  # pinch just released


class PinchTracker:
    """Stateful pinch tracker with hysteresis.

    Args:
        down_threshold: Aperture below which a pinch is detected.
        up_threshold:   Aperture above which the pinch is released.
                        Must be > ``down_threshold`` to create a hysteresis band.
    """

    def __init__(
        self,
        down_threshold: float = 0.22,
        up_threshold: float = 0.32,
    ) -> None:
        if up_threshold <= down_threshold:
            raise ValueError(
                f"up_threshold ({up_threshold}) must be > down_threshold ({down_threshold})"
            )
        self._down_thr = down_threshold
        self._up_thr = up_threshold
        self._pinching = False
        self._pinch_start_time: float | None = None

    @property
    def is_pinching(self) -> bool:
        """True while a pinch is currently in progress."""
        return self._pinching

    @property
    def pinch_start_time(self) -> float | None:
        """Monotonic timestamp of when the current pinch started, or None."""
        return self._pinch_start_time

    def update(self, aperture: float) -> PinchEvent:
        """Feed the current thumb-index normalized aperture and return an event.

        Args:
            aperture: Normalized pinch aperture from :class:`FeatureSet`
                      (``thumb_index_distance``).  Values in [0.0, 1.0].

        Returns:
            A :class:`PinchEvent` value.
        """
        if not self._pinching and aperture < self._down_thr:
            self._pinching = True
            self._pinch_start_time = time.monotonic()
            return PinchEvent.DOWN

        if self._pinching and aperture > self._up_thr:
            self._pinching = False
            self._pinch_start_time = None
            return PinchEvent.UP

        if self._pinching:
            return PinchEvent.HOLD

        return PinchEvent.NONE

    def reset(self) -> None:
        """Force-clear pinch state (call after pipeline restarts)."""
        self._pinching = False
        self._pinch_start_time = None
