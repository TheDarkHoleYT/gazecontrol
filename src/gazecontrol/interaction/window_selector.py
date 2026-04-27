"""WindowSelector — enumerate visible desktop windows and hit-test a screen point.

Coordinate space: the point is expected in **virtual-desktop pixels** —
the same coordinate system as ``GetWindowRect`` on a DPI-aware process.
Ensure the process is DPI-aware (``SetProcessDpiAwarenessContext`` called at
startup) so that ``GetWindowRect`` returns physical rather than virtualised
coordinates.
"""

from __future__ import annotations

import contextlib
import ctypes
import time
from typing import Any

try:
    import win32con
    import win32gui

    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

_DWMWA_CLOAKED = 14
_WS_EX_TOOLWINDOW = 0x00000080


def _is_cloaked(hwnd: int) -> bool:
    """Return True when the DWM reports the window as cloaked (invisible)."""
    try:
        cloaked = ctypes.c_int(0)
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            ctypes.c_int(hwnd),
            ctypes.c_int(_DWMWA_CLOAKED),
            ctypes.byref(cloaked),
            ctypes.sizeof(cloaked),
        )
        return cloaked.value != 0
    except Exception:
        return False


class WindowSelector:
    """Enumerate visible windows and find the one under a screen-space point.

    Filtering applied during enumeration:

    - ``IsWindowVisible`` must return True.
    - Window must have a non-empty title.
    - Window must be its own top-level root (``GetAncestor(GA_ROOT) == hwnd``).
    - Window must not have the ``WS_EX_TOOLWINDOW`` extended style.
    - Window must not be DWM-cloaked.
    """

    def __init__(self, cache_interval: float = 0.5) -> None:
        self._cache_interval = cache_interval
        self._cache: list[dict[str, Any]] = []
        self._last_update: float = 0.0

    def invalidate(self) -> None:
        """Force the next :meth:`find_window` call to re-enumerate all windows."""
        self._last_update = 0.0

    def _refresh_cache(self) -> None:
        now = time.monotonic()
        if now - self._last_update < self._cache_interval and self._cache:
            return
        self._last_update = now
        if not HAS_WIN32:
            self._cache = []
            return
        windows: list[dict[str, Any]] = []

        def enum_callback(hwnd: int, _: Any) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return
            try:
                root = win32gui.GetAncestor(hwnd, 2)
                if root != hwnd:
                    return
            except Exception:
                return
            try:
                ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                if ex_style & _WS_EX_TOOLWINDOW:
                    return
            except Exception:
                pass
            if _is_cloaked(hwnd):
                return
            try:
                rect = win32gui.GetWindowRect(hwnd)
            except Exception:
                return
            x, y, x2, y2 = rect
            w = x2 - x
            h = y2 - y
            if w <= 0 or h <= 0:
                return
            windows.append({"hwnd": hwnd, "title": title, "rect": (x, y, w, h)})

        with contextlib.suppress(Exception):
            win32gui.EnumWindows(enum_callback, None)
        self._cache = windows

    def find_window(self, point: tuple[float, float] | None) -> dict[str, Any] | None:
        """Return the topmost visible window under *point*, or None.

        Args:
            point: Position in virtual-desktop pixel coordinates.
        """
        if not point:
            return None
        self._refresh_cache()
        gx, gy = point[0], point[1]
        for win in self._cache:
            x, y, w, h = win["rect"]
            # Half-open interval: the right/bottom edges belong to the next
            # pixel (the standard pixel hit-test convention).  Avoids two
            # adjacent windows both claiming an edge-pixel hit.
            if x <= gx < x + w and y <= gy < y + h:
                return win
        return None
