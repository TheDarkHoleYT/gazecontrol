"""Tests for WindowsManager — Win32 error handling (mocked).

All Win32 API calls are mocked so tests run cross-platform.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


def _make_mock_win32():
    win32gui = MagicMock()
    win32con = MagicMock()
    win32api = MagicMock()
    win32con.WM_CLOSE = 0x0010
    win32con.SW_MINIMIZE = 6
    win32con.SW_MAXIMIZE = 3
    win32con.WM_MOUSEWHEEL = 0x020A
    return win32gui, win32con, win32api


@pytest.fixture
def manager_mocks(monkeypatch):
    """Patch win32 modules and return (manager, win32gui_mock, win32api_mock)."""
    win32gui, win32con, win32api = _make_mock_win32()
    monkeypatch.setitem(sys.modules, "win32gui", win32gui)
    monkeypatch.setitem(sys.modules, "win32con", win32con)
    monkeypatch.setitem(sys.modules, "win32api", win32api)

    # Reload so the module picks up mocked imports.

    if "gazecontrol.window_manager.windows_mgr" in sys.modules:
        del sys.modules["gazecontrol.window_manager.windows_mgr"]

    from gazecontrol.window_manager.windows_mgr import WindowsManager

    mgr = WindowsManager()
    mgr.HAS_WIN32 = True  # ensure the branches are taken
    return mgr, win32gui, win32api


def test_move_window_calls_set_window_pos(manager_mocks):
    """move_window() must call SetWindowPos (not MoveWindow) to avoid repaint flicker."""
    mgr, win32gui, _ = manager_mocks
    win32gui.GetWindowRect.return_value = (100, 100, 500, 400)
    mgr.move_window(1001, 200, 300)
    win32gui.SetWindowPos.assert_called_once()
    win32gui.MoveWindow.assert_not_called()


def test_close_window_posts_message(manager_mocks):
    mgr, win32gui, _ = manager_mocks
    mgr.close_window(1001)
    win32gui.PostMessage.assert_called_once()


def test_execute_dispatches_drag(manager_mocks):
    """DRAG action must be forwarded to the base class execute() which calls move_window."""
    mgr, win32gui, _ = manager_mocks
    win32gui.GetWindowRect.return_value = (100, 100, 500, 400)
    action = {
        "type": "DRAG",
        "window": {"hwnd": 1001},
        "data": {"phase": "move", "delta": (10, 20), "start_rect": (100, 100, 400, 300)},
    }
    mgr.execute(action)
    # Now uses SetWindowPos instead of MoveWindow.
    win32gui.SetWindowPos.assert_called()
    win32gui.MoveWindow.assert_not_called()


def test_execute_scroll_up(manager_mocks, monkeypatch):
    """SCROLL_UP must use SendInput (not PostMessage) to work in Chromium/Electron."""
    mgr, _win32gui, _ = manager_mocks
    import ctypes

    mock_send_input = MagicMock(return_value=1)
    monkeypatch.setattr(ctypes.windll.user32, "SendInput", mock_send_input)
    action = {"type": "SCROLL_UP", "window": {"hwnd": 1001}, "data": {}}
    mgr.execute(action)
    mock_send_input.assert_called_once()


def test_win32_error_logged_not_raised(manager_mocks):
    """Exceptions in Win32 calls should be logged, not propagated."""
    mgr, win32gui, _ = manager_mocks
    win32gui.SetWindowPos.side_effect = OSError("access denied")
    win32gui.GetWindowRect.return_value = (0, 0, 400, 300)
    # Should not raise.
    mgr.move_window(1001, 0, 0)
