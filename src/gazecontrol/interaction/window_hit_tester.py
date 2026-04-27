"""WindowHitTester — point-in-window lookup for the hand interaction layer.

Wraps ``WindowSelector`` (Win32 EnumWindows-based enumeration) and adapts
its dict-based output into typed :class:`~gazecontrol.interaction.types.HoveredWindow`
instances.

The underlying ``WindowSelector`` caches the window list for 0.5 s to avoid
hammering the OS with ``EnumWindows`` on every pipeline tick (typically 30 fps).
"""

from __future__ import annotations

from gazecontrol.interaction.types import HoveredWindow


class WindowHitTester:
    """Find the topmost visible window under a screen-space point.

    Args:
        cache_interval: How often to re-enumerate windows (seconds).
                        Passed through to :class:`WindowSelector`.
    """

    def __init__(self, cache_interval: float = 0.5) -> None:
        # Import lazily so the class is constructable without Win32 in tests.
        from gazecontrol.interaction.window_selector import WindowSelector

        self._selector = WindowSelector(cache_interval=cache_interval)

    def at(self, point: tuple[int, int] | None) -> HoveredWindow | None:
        """Return the window under *point*, or ``None``.

        Args:
            point: Screen coordinate ``(x, y)`` in virtual-desktop pixels.
        """
        if point is None:
            return None
        raw = self._selector.find_window(point)
        if raw is None:
            return None
        return HoveredWindow(
            hwnd=raw["hwnd"],
            rect=tuple(raw["rect"]),
            title=raw.get("title", ""),
        )

    def invalidate(self) -> None:
        """Force the next :meth:`at` call to re-enumerate windows."""
        self._selector.invalidate()
