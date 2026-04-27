"""DeadZone — circular dead-zone with hysteresis for pointer stabilisation.

Suppresses micro-jitter when the hand is nearly stationary.  Two radii
create a hysteresis band that prevents rapid in/out toggling at the boundary:

- If the pointer moves **within** ``radius_px`` of the last locked position,
  it stays locked (jitter suppressed).
- Once movement **exceeds** ``hysteresis_px``, the lock is released and the
  pointer moves freely until it becomes slow again.

Usage::

    dz = DeadZone(radius_px=4, hysteresis_px=8)

    for sx, sy in raw_pointer_stream:
        sx_stable, sy_stable = dz.apply(sx, sy)
"""

from __future__ import annotations

import math


class DeadZone:
    """Circular dead-zone with hysteresis for (x, y) pixel coordinates.

    Args:
        radius_px:     Inner radius — movements smaller than this are suppressed.
        hysteresis_px: Outer radius — the lock is released only when movement
                       exceeds this distance from the locked anchor.
                       Must be ≥ ``radius_px``.

    The filter is stateful — one instance per pointer stream.
    """

    def __init__(self, radius_px: float = 3.0, hysteresis_px: float = 6.0) -> None:
        self._r_in = float(radius_px)
        self._r_out = float(max(hysteresis_px, radius_px))

        # Current anchor (locked position).  None = filter not yet initialised.
        self._anchor_x: float | None = None
        self._anchor_y: float | None = None
        self._locked = False

    # ------------------------------------------------------------------

    def apply(self, x: float, y: float) -> tuple[float, float]:
        """Apply the dead-zone and return the (possibly stabilised) position.

        Args:
            x: Raw pointer x coordinate (pixels).
            y: Raw pointer y coordinate (pixels).

        Returns:
            ``(sx, sy)`` — stabilised coordinates.
        """
        if self._anchor_x is None or self._anchor_y is None:
            # First call: initialise anchor at current position.
            self._anchor_x = x
            self._anchor_y = y
            self._locked = True
            return x, y

        ax = self._anchor_x
        ay = self._anchor_y
        dist = math.sqrt((x - ax) ** 2 + (y - ay) ** 2)

        if self._locked:
            if dist > self._r_out:
                # Hand moved past hysteresis boundary → unlock, pass through.
                self._locked = False
                self._anchor_x = x
                self._anchor_y = y
                return x, y
            # Still within dead-zone → return locked anchor.
            return ax, ay
        else:
            # Unlocked: update anchor continuously.
            self._anchor_x = x
            self._anchor_y = y
            if dist < self._r_in:
                # Settled back inside inner radius → re-lock.
                self._locked = True
            return x, y

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset state (call when hand tracking is lost)."""
        self._anchor_x = None
        self._anchor_y = None
        self._locked = False
