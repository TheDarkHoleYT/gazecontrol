"""Tests for BaseWindowManager.execute dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

from gazecontrol.window_manager.base import BaseWindowManager


class _ConcreteManager(BaseWindowManager):
    """Minimal concrete implementation for testing base dispatch."""

    def move_window(self, hwnd, x: int, y: int) -> None: ...
    def resize_window(self, hwnd, x: int, y: int, w: int, h: int) -> None: ...
    def close_window(self, hwnd) -> None: ...
    def minimize_window(self, hwnd) -> None: ...
    def maximize_window(self, hwnd) -> None: ...
    def bring_to_front(self, hwnd) -> None: ...


def _patched_manager() -> _ConcreteManager:
    """Return a _ConcreteManager with all primitives replaced by MagicMocks."""
    mgr = _ConcreteManager()
    mgr.move_window = MagicMock()  # type: ignore[method-assign]
    mgr.resize_window = MagicMock()  # type: ignore[method-assign]
    mgr.close_window = MagicMock()  # type: ignore[method-assign]
    mgr.minimize_window = MagicMock()  # type: ignore[method-assign]
    mgr.maximize_window = MagicMock()  # type: ignore[method-assign]
    mgr.bring_to_front = MagicMock()  # type: ignore[method-assign]
    return mgr


def test_execute_drag_calls_move():
    mgr = _patched_manager()
    action = {
        "type": "DRAG",
        "window": {"hwnd": 100},
        "data": {"phase": "move", "delta": (10, 20), "start_rect": (0, 0, 300, 200)},
    }
    mgr.execute(action)
    mgr.move_window.assert_called_once_with(100, 10, 20)


def test_execute_drag_start_no_call():
    mgr = _patched_manager()
    action = {
        "type": "DRAG",
        "window": {"hwnd": 100},
        "data": {"phase": "start", "start_rect": (0, 0, 300, 200)},
    }
    mgr.execute(action)
    mgr.move_window.assert_not_called()


def test_execute_resize_calls_resize():
    mgr = _patched_manager()
    action = {
        "type": "RESIZE",
        "window": {"hwnd": 100},
        "data": {"phase": "move", "delta": (50, 30), "start_rect": (0, 0, 300, 200)},
    }
    mgr.execute(action)
    mgr.resize_window.assert_called_once_with(100, 0, 0, 350, 230)


def test_execute_close():
    mgr = _patched_manager()
    mgr.execute({"type": "CLOSE", "window": {"hwnd": 100}, "data": {}})
    mgr.close_window.assert_called_once_with(100)


def test_execute_minimize():
    mgr = _patched_manager()
    mgr.execute({"type": "MINIMIZE", "window": {"hwnd": 100}, "data": {}})
    mgr.minimize_window.assert_called_once_with(100)


def test_execute_maximize():
    mgr = _patched_manager()
    mgr.execute({"type": "MAXIMIZE", "window": {"hwnd": 100}, "data": {}})
    mgr.maximize_window.assert_called_once_with(100)


def test_execute_bring_front():
    mgr = _patched_manager()
    mgr.execute({"type": "BRING_FRONT", "window": {"hwnd": 100}, "data": {}})
    mgr.bring_to_front.assert_called_once_with(100)


def test_execute_no_hwnd_does_nothing():
    mgr = _patched_manager()
    mgr.execute({"type": "CLOSE", "window": {}, "data": {}})
    mgr.close_window.assert_not_called()


def test_execute_scroll_not_handled_by_base():
    """SCROLL_UP and SCROLL_DOWN are silently ignored by base — no crash."""
    mgr = _patched_manager()
    mgr.execute({"type": "SCROLL_UP", "window": {"hwnd": 100}, "data": {}})
    # No call to any primitive; no exception.


def test_execute_resize_min_size():
    """Resize must clamp width/height to minimum values."""
    mgr = _patched_manager()
    action = {
        "type": "RESIZE",
        "window": {"hwnd": 100},
        "data": {"phase": "move", "delta": (-5000, -5000), "start_rect": (0, 0, 300, 200)},
    }
    mgr.execute(action)
    _, _, _, new_w, new_h = mgr.resize_window.call_args[0]
    assert new_w >= 100
    assert new_h >= 60
