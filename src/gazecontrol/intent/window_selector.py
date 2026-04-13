"""WindowSelector — find the desktop window under the gaze point."""
from __future__ import annotations

import contextlib
import time
from typing import Any

try:
    import win32gui
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


class WindowSelector:
    """Enumerate visible windows and find the one under the current gaze point."""

    def __init__(self, cache_interval: float = 0.5) -> None:
        self._cache_interval = cache_interval
        self._cache: list[dict] = []
        self._last_update: float = 0

    def _refresh_cache(self) -> None:
        now = time.time()
        if now - self._last_update < self._cache_interval and self._cache:
            return
        self._last_update = now
        if not HAS_WIN32:
            self._cache = []
            return
        windows: list[dict] = []

        def enum_callback(hwnd: int, _: Any) -> None:  # noqa: ANN401
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if not title:
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
            windows.append({
                'hwnd': hwnd,
                'title': title,
                'rect': (x, y, w, h),
            })

        with contextlib.suppress(Exception):
            win32gui.EnumWindows(enum_callback, None)
        self._cache = windows

    def find_window(self, gaze_point: tuple[float, float] | None) -> dict | None:
        """Return the topmost visible window under *gaze_point*, or None."""
        if not gaze_point:
            return None
        self._refresh_cache()
        gx, gy = gaze_point[0], gaze_point[1]
        for win in self._cache:
            x, y, w, h = win['rect']
            if x <= gx <= x + w and y <= gy <= y + h:
                return win
        return None
