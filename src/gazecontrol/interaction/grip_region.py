"""Grip-region helper — detects whether a screen point is inside the resize corner."""

from __future__ import annotations


def is_in_resize_grip(
    rect: tuple[int, int, int, int],
    point: tuple[int, int],
    ratio: float = 0.18,
) -> bool:
    """Return ``True`` when *point* falls inside the bottom-right resize grip.

    The resize grip is defined as the rectangular region occupying the last
    ``ratio * 100 %`` of both the width and the height of the window, measured
    from the bottom-right corner.

    Args:
        rect:  Window bounding box ``(x, y, width, height)`` in screen pixels.
        point: Test point ``(px, py)`` in screen pixels.
        ratio: Fraction of each dimension that constitutes the grip zone.
               Default ``0.18`` (18 %).

    Returns:
        ``True`` if ``point`` is inside the grip zone.
    """
    x, y, w, h = rect
    px, py = point
    grip_x = x + int(w * (1.0 - ratio))
    grip_y = y + int(h * (1.0 - ratio))
    return px >= grip_x and py >= grip_y
