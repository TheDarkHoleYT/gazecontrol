"""Windows-specific window manager — Win32 API implementation."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
from typing import Any

from gazecontrol.errors import GazeControlError
from gazecontrol.window_manager.base import BaseWindowManager

logger = logging.getLogger(__name__)

try:
    import win32api
    import win32con
    import win32gui

    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

# SendInput constants.
_INPUT_MOUSE = 0
_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_ABSOLUTE = 0x8000


class WindowManagerError(GazeControlError, OSError):
    """Raised when a Win32 window operation fails."""


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", _MOUSEINPUT),
    ]


def _send_mouse_input(flags: int, dx: int = 0, dy: int = 0, mouse_data: int = 0) -> None:
    """Low-level helper: build and dispatch a single SendInput mouse event."""
    inp = _INPUT()
    inp.type = _INPUT_MOUSE
    inp.mi.dx = dx
    inp.mi.dy = dy
    inp.mi.mouseData = ctypes.c_ulong(mouse_data & 0xFFFFFFFF)
    inp.mi.dwFlags = flags
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


class WindowsManager(BaseWindowManager):
    """Win32 implementation of BaseWindowManager.

    Implements primitive window operations plus synthetic mouse input
    (click, scroll) and screen-space hit-testing.
    """

    # ------------------------------------------------------------------
    # Window primitives
    # ------------------------------------------------------------------

    def move_window(self, hwnd: int, x: int, y: int) -> None:
        if not HAS_WIN32:
            return
        try:
            rect = win32gui.GetWindowRect(hwnd)
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            win32gui.SetWindowPos(
                hwnd,
                0,
                x,
                y,
                w,
                h,
                win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE | win32con.SWP_ASYNCWINDOWPOS,
            )
        except Exception as exc:
            logger.warning("move_window(hwnd=%d, x=%d, y=%d) failed: %s", hwnd, x, y, exc)

    def resize_window(self, hwnd: int, x: int, y: int, w: int, h: int) -> None:
        if not HAS_WIN32:
            return
        try:
            win32gui.SetWindowPos(
                hwnd,
                0,
                x,
                y,
                w,
                h,
                win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE,
            )
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
            win32gui.ShowWindowAsync(hwnd, win32con.SW_MINIMIZE)
        except Exception as exc:
            logger.warning("minimize_window(hwnd=%d) failed: %s", hwnd, exc)

    def maximize_window(self, hwnd: int) -> None:
        if not HAS_WIN32:
            return
        try:
            win32gui.ShowWindowAsync(hwnd, win32con.SW_MAXIMIZE)
        except Exception as exc:
            logger.warning("maximize_window(hwnd=%d) failed: %s", hwnd, exc)

    def bring_to_front(self, hwnd: int) -> None:
        """Bring *hwnd* to the foreground with AttachThreadInput fallback."""
        if not HAS_WIN32:
            return
        try:
            fg_hwnd = win32gui.GetForegroundWindow()
            fg_tid = win32api.GetWindowThreadProcessId(fg_hwnd)[0]
            our_tid = win32api.GetCurrentThreadId()

            if fg_tid != our_tid:
                ctypes.windll.user32.AttachThreadInput(our_tid, fg_tid, True)
                try:
                    result = win32gui.SetForegroundWindow(hwnd)
                finally:
                    ctypes.windll.user32.AttachThreadInput(our_tid, fg_tid, False)
            else:
                result = win32gui.SetForegroundWindow(hwnd)

            if not result:
                logger.warning(
                    "bring_to_front(hwnd=%d): SetForegroundWindow returned 0.",
                    hwnd,
                )
        except Exception as exc:
            logger.warning("bring_to_front(hwnd=%d) failed: %s", hwnd, exc)

    # ------------------------------------------------------------------
    # Synthetic mouse input
    # ------------------------------------------------------------------

    def click_at(self, x: int, y: int, button: str = "left") -> None:
        """Synthesise a mouse click at screen coordinate *(x, y)*.

        Moves the OS cursor to the target position, then sends a
        button-down + button-up pair via SendInput.

        Args:
            x, y:   Screen coordinates in virtual-desktop pixels.
            button: ``"left"`` or ``"right"``.
        """
        try:
            ctypes.windll.user32.SetCursorPos(x, y)
            if button == "right":
                _send_mouse_input(_MOUSEEVENTF_RIGHTDOWN)
                _send_mouse_input(_MOUSEEVENTF_RIGHTUP)
            else:
                _send_mouse_input(_MOUSEEVENTF_LEFTDOWN)
                _send_mouse_input(_MOUSEEVENTF_LEFTUP)
        except Exception as exc:
            logger.warning("click_at(%d, %d) failed: %s", x, y, exc)

    def double_click_at(self, x: int, y: int) -> None:
        """Synthesise a double left-click at *(x, y)*."""
        self.click_at(x, y)
        self.click_at(x, y)

    def scroll_at(self, x: int, y: int, delta: int = 120) -> None:
        """Synthesise a mouse-wheel scroll at screen coordinate *(x, y)*.

        Args:
            x, y:  Target screen coordinates (cursor is moved here first).
            delta: Positive = scroll up, negative = scroll down.
                   Typically ±120 per notch.
        """
        try:
            ctypes.windll.user32.SetCursorPos(x, y)
            _send_mouse_input(_MOUSEEVENTF_WHEEL, mouse_data=delta)
        except Exception as exc:
            logger.warning("scroll_at(%d, %d, delta=%d) failed: %s", x, y, delta, exc)

    def scroll(self, hwnd: int, delta: int) -> None:
        """Legacy scroll — delegates to :meth:`scroll_at` at current cursor pos.

        ``hwnd`` is unused — scroll targets the window under the current cursor.
        """
        try:
            _send_mouse_input(_MOUSEEVENTF_WHEEL, mouse_data=delta)
        except Exception as exc:
            logger.warning("scroll(delta=%d) failed: %s", delta, exc)

    def get_window_rect(self, hwnd: int) -> tuple[int, int, int, int] | None:
        """Return ``(x, y, width, height)`` for *hwnd*, or ``None`` on failure."""
        if not HAS_WIN32:
            return None
        try:
            x, y, x2, y2 = win32gui.GetWindowRect(hwnd)
            return (x, y, x2 - x, y2 - y)
        except Exception as exc:
            logger.warning("get_window_rect(hwnd=%d) failed: %s", hwnd, exc)
            return None

    # ------------------------------------------------------------------
    # Override execute to handle old dict-style dispatch (still needed
    # for any callers that use the base-class execute path).
    # ------------------------------------------------------------------

    def execute(self, action: dict[str, Any]) -> None:
        """Dispatch action dict; handles SCROLL_* here, delegates rest to base."""
        action_type = action.get("type")
        if action_type in ("SCROLL_UP", "SCROLL_DOWN"):
            hwnd = (action.get("window") or {}).get("hwnd")
            if hwnd:
                delta = 120 if action_type == "SCROLL_UP" else -120
                self.scroll(hwnd, delta)
            return
        super().execute(action)
