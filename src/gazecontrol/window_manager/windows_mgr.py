"""Windows-specific window manager — Win32 API implementation."""
from __future__ import annotations

import ctypes
import logging

from gazecontrol.window_manager.base import BaseWindowManager

logger = logging.getLogger(__name__)

try:
    import win32api
    import win32con
    import win32gui

    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


class WindowManagerError(OSError):
    """Raised when a Win32 window operation fails."""


def _win32_call(operation: str, fn: object, *args: object) -> None:
    """Execute a Win32 call, logging warnings on pywintypes.error."""
    if not HAS_WIN32:
        return
    try:
        fn(*args)  # type: ignore[operator]
    except Exception as exc:  # pywintypes.error inherits from Exception
        logger.warning("WindowsManager.%s failed: %s", operation, exc)


class WindowsManager(BaseWindowManager):
    """Win32 implementation of BaseWindowManager.

    Only implements the primitive operations (move, resize, close, minimize,
    maximize, bring_to_front, scroll).  All dispatch logic lives in the base
    class ``execute()`` method.
    """

    def move_window(self, hwnd: int, x: int, y: int) -> None:
        if not HAS_WIN32:
            return
        try:
            rect = win32gui.GetWindowRect(hwnd)
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            win32gui.MoveWindow(hwnd, x, y, w, h, True)
        except Exception as exc:
            logger.warning("move_window(hwnd=%d, x=%d, y=%d) failed: %s", hwnd, x, y, exc)

    def resize_window(self, hwnd: int, x: int, y: int, w: int, h: int) -> None:
        if not HAS_WIN32:
            return
        try:
            win32gui.MoveWindow(hwnd, x, y, w, h, True)
        except Exception as exc:
            logger.warning("resize_window(hwnd=%d) failed: %s", hwnd, exc)

    def close_window(self, hwnd: int) -> None:
        if not HAS_WIN32:
            return
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        except Exception as exc:
            logger.warning("close_window(hwnd=%d) failed: %s", hwnd, exc)

    def minimize_window(self, hwnd: int) -> None:
        if not HAS_WIN32:
            return
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        except Exception as exc:
            logger.warning("minimize_window(hwnd=%d) failed: %s", hwnd, exc)

    def maximize_window(self, hwnd: int) -> None:
        if not HAS_WIN32:
            return
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        except Exception as exc:
            logger.warning("maximize_window(hwnd=%d) failed: %s", hwnd, exc)

    def bring_to_front(self, hwnd: int) -> None:
        if not HAS_WIN32:
            return
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception as exc:
            logger.warning("bring_to_front(hwnd=%d) failed: %s", hwnd, exc)

    def scroll(self, hwnd: int, delta: int) -> None:
        """Send WM_MOUSEWHEEL to a window.

        Args:
            hwnd:  Target window handle.
            delta: Positive = scroll up, negative = scroll down.
                   Typically ±120 per notch.
        """
        if not HAS_WIN32:
            return
        try:
            # Encode delta as unsigned 16-bit high word of WPARAM.
            high_word = ctypes.c_int16(delta).value & 0xFFFF
            wparam = high_word << 16
            win32api.PostMessage(hwnd, win32con.WM_MOUSEWHEEL, wparam, 0)
        except Exception as exc:
            logger.warning("scroll(hwnd=%d, delta=%d) failed: %s", hwnd, delta, exc)

    def execute(self, action: dict) -> None:
        """Dispatch action dict; handles SCROLL_* here, delegates rest to base class."""
        action_type = action.get("type")
        if action_type in ("SCROLL_UP", "SCROLL_DOWN"):
            hwnd = action.get("window", {}).get("hwnd")
            if hwnd:
                delta = 120 if action_type == "SCROLL_UP" else -120
                self.scroll(hwnd, delta)
            return
        super().execute(action)
