"""WindowSelector — pure hit-test logic exercised against a synthetic cache."""

from __future__ import annotations

from gazecontrol.interaction.window_selector import WindowSelector


def _make_selector(windows: list[dict]) -> WindowSelector:
    """Build a WindowSelector with a pre-populated cache (skip enumeration)."""
    sel = WindowSelector()
    sel._cache = windows  # bypass _refresh_cache by setting cache directly
    sel._last_update = 1e18  # far future → cache treated as fresh
    return sel


def test_find_window_returns_none_for_no_point():
    sel = _make_selector([{"hwnd": 1, "title": "A", "rect": (0, 0, 100, 100)}])
    assert sel.find_window(None) is None


def test_find_window_inside_rect():
    sel = _make_selector([{"hwnd": 1, "title": "A", "rect": (0, 0, 100, 100)}])
    win = sel.find_window((50.0, 50.0))
    assert win is not None
    assert win["hwnd"] == 1


def test_find_window_top_left_corner_inclusive():
    """The top-left corner (x, y) belongs to the window."""
    sel = _make_selector([{"hwnd": 1, "title": "A", "rect": (10, 20, 100, 100)}])
    win = sel.find_window((10.0, 20.0))
    assert win is not None and win["hwnd"] == 1


def test_find_window_right_bottom_edges_exclusive():
    """Regression for BUG-006: right and bottom edges are EXCLUSIVE.

    A point exactly on ``x + w`` or ``y + h`` belongs to the next pixel —
    matching the standard pixel hit-test convention.  Two adjacent windows
    must not both claim the shared edge."""
    sel = _make_selector([{"hwnd": 1, "title": "A", "rect": (0, 0, 100, 100)}])
    # Right edge is x=100; that pixel is outside the window.
    assert sel.find_window((100.0, 50.0)) is None
    assert sel.find_window((50.0, 100.0)) is None
    assert sel.find_window((100.0, 100.0)) is None


def test_two_adjacent_windows_no_double_match_on_edge():
    """Two windows touching at x=100 must not both claim a hit at x=100."""
    sel = _make_selector(
        [
            {"hwnd": 1, "title": "Left", "rect": (0, 0, 100, 100)},
            {"hwnd": 2, "title": "Right", "rect": (100, 0, 100, 100)},
        ]
    )
    win = sel.find_window((100.0, 50.0))
    assert win is not None
    # Only the right window owns x=100 with the half-open convention.
    assert win["hwnd"] == 2
