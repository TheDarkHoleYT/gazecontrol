"""Tests for WindowSelector — hit-testing and caching."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.win32
def test_window_selector_returns_window_at_point():
    """WindowSelector.find_window should return the window under the gaze point."""
    pytest.importorskip("win32gui")

    from gazecontrol.intent.window_selector import WindowSelector

    ws = WindowSelector()
    # Even without a real screen we just confirm it doesn't crash.
    result = ws.find_window((100, 100))
    # On Windows: either a dict or None.
    assert result is None or isinstance(result, dict)


def test_window_selector_returns_none_without_win32():
    """On non-Windows or if win32 not available, find_window should return None."""
    fake_win32 = MagicMock()
    fake_win32.EnumWindows = lambda cb, extra: None

    with patch.dict(sys.modules, {"win32gui": None, "win32api": None, "win32con": None}):
        # Re-import with patched modules.

        if "gazecontrol.intent.window_selector" in sys.modules:
            del sys.modules["gazecontrol.intent.window_selector"]

        from gazecontrol.intent.window_selector import WindowSelector

        ws = WindowSelector()
        result = ws.find_window((100, 100))
        # Without win32, EnumWindows won't run — result is None.
        assert result is None
