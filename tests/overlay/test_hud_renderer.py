"""Tests for HUDRenderer — pure-logic paths (no real Qt painter required)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def renderer():
    from gazecontrol.overlay.hud_renderer import HUDRenderer

    return HUDRenderer()


def test_render_no_crash_when_pyqt_absent(renderer):
    """render() must be a no-op (not raise) when HAS_PYQT is False."""
    import gazecontrol.overlay.hud_renderer as mod

    orig = mod.HAS_PYQT
    try:
        mod.HAS_PYQT = False
        painter = MagicMock()
        renderer.render(painter, {"fingertip_screen": (100, 200), "state": "IDLE"})
        painter.assert_not_called()
    finally:
        mod.HAS_PYQT = orig


def test_render_calls_draw_pointer_when_fingertip_present(renderer):
    renderer._draw_pointer = MagicMock()
    renderer._draw_state_label = MagicMock()
    renderer._draw_window_border = MagicMock()
    renderer._draw_resize_grip_hint = MagicMock()

    import gazecontrol.overlay.hud_renderer as mod

    if not mod.HAS_PYQT:
        pytest.skip("PyQt6 not installed")

    painter = MagicMock()
    renderer.render(
        painter,
        {
            "fingertip_screen": (500, 300),
            "state": "IDLE",
            "hovered_window": None,
            "gesture_id": None,
        },
    )
    from unittest.mock import ANY

    renderer._draw_pointer.assert_called_once_with(painter, (500, 300), "IDLE", ANY)


def test_render_skips_pointer_when_no_fingertip(renderer):
    renderer._draw_pointer = MagicMock()
    renderer._draw_state_label = MagicMock()

    import gazecontrol.overlay.hud_renderer as mod

    if not mod.HAS_PYQT:
        pytest.skip("PyQt6 not installed")

    painter = MagicMock()
    renderer.render(painter, {"fingertip_screen": None, "state": "IDLE"})
    renderer._draw_pointer.assert_not_called()


def test_render_draws_window_border_in_non_idle_state(renderer):
    renderer._draw_window_border = MagicMock()
    renderer._draw_resize_grip_hint = MagicMock()
    renderer._draw_state_label = MagicMock()
    renderer._draw_pointer = MagicMock()

    import gazecontrol.overlay.hud_renderer as mod

    if not mod.HAS_PYQT:
        pytest.skip("PyQt6 not installed")

    from gazecontrol.interaction.types import HoveredWindow

    hw = HoveredWindow(hwnd=1, rect=(0, 0, 800, 600), title="Test")
    painter = MagicMock()
    renderer.render(
        painter,
        {
            "fingertip_screen": (100, 100),
            "state": "DRAGGING",
            "hovered_window": hw,
            "gesture_id": None,
        },
    )
    renderer._draw_window_border.assert_called_once()


def test_render_calls_status_bar(renderer):
    """render() must always call _draw_status_bar regardless of hand state."""
    renderer._draw_status_bar = MagicMock()
    renderer._draw_heartbeat = MagicMock()
    renderer._draw_state_label = MagicMock()

    import gazecontrol.overlay.hud_renderer as mod

    if not mod.HAS_PYQT:
        pytest.skip("PyQt6 not installed")

    painter = MagicMock()
    renderer.render(painter, {"fingertip_screen": None, "state": "IDLE", "capture_ok": True})
    renderer._draw_status_bar.assert_called_once_with(painter, capture_ok=True, hand_detected=False)


def test_render_shows_no_hand_banner_when_no_fingertip(renderer):
    renderer._draw_no_hand = MagicMock()
    renderer._draw_heartbeat = MagicMock()
    renderer._draw_status_bar = MagicMock()
    renderer._draw_state_label = MagicMock()

    import gazecontrol.overlay.hud_renderer as mod

    if not mod.HAS_PYQT:
        pytest.skip("PyQt6 not installed")

    painter = MagicMock()
    renderer.render(painter, {"fingertip_screen": None, "state": "IDLE", "capture_ok": True})
    renderer._draw_no_hand.assert_called_once_with(painter)


def test_render_no_hand_banner_when_capture_fails(renderer):
    """When capture_ok is False render() must not try to draw no-hand banner."""
    renderer._draw_no_hand = MagicMock()
    renderer._draw_heartbeat = MagicMock()
    renderer._draw_status_bar = MagicMock()
    renderer._draw_state_label = MagicMock()

    import gazecontrol.overlay.hud_renderer as mod

    if not mod.HAS_PYQT:
        pytest.skip("PyQt6 not installed")

    painter = MagicMock()
    renderer.render(painter, {"fingertip_screen": None, "state": "IDLE", "capture_ok": False})
    renderer._draw_no_hand.assert_not_called()


def test_render_draws_coords_near_pointer(renderer):
    renderer._draw_pointer = MagicMock()
    renderer._draw_coords = MagicMock()
    renderer._draw_heartbeat = MagicMock()
    renderer._draw_status_bar = MagicMock()
    renderer._draw_state_label = MagicMock()

    import gazecontrol.overlay.hud_renderer as mod

    if not mod.HAS_PYQT:
        pytest.skip("PyQt6 not installed")

    painter = MagicMock()
    renderer.render(
        painter,
        {
            "fingertip_screen": (640, 360),
            "state": "IDLE",
            "capture_ok": True,
        },
    )
    renderer._draw_coords.assert_called_once_with(painter, (640, 360))


def test_render_does_not_draw_border_in_idle(renderer):
    renderer._draw_window_border = MagicMock()
    renderer._draw_state_label = MagicMock()
    renderer._draw_pointer = MagicMock()

    import gazecontrol.overlay.hud_renderer as mod

    if not mod.HAS_PYQT:
        pytest.skip("PyQt6 not installed")

    from gazecontrol.interaction.types import HoveredWindow

    hw = HoveredWindow(hwnd=1, rect=(0, 0, 800, 600), title="Test")
    painter = MagicMock()
    renderer.render(
        painter,
        {
            "fingertip_screen": (100, 100),
            "state": "IDLE",
            "hovered_window": hw,
            "gesture_id": None,
        },
    )
    renderer._draw_window_border.assert_not_called()
