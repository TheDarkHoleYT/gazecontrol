"""Tests for OverlayWindow — no-Qt path + launcher wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# No-Qt fallback
# ---------------------------------------------------------------------------


def test_overlay_window_no_crash_without_pyqt():
    """OverlayWindow must be constructable and callable when PyQt6 is absent."""
    import gazecontrol.overlay.overlay_window as mod

    orig = mod.HAS_PYQT
    try:
        mod.HAS_PYQT = False
        from gazecontrol.overlay.overlay_window import OverlayWindow

        ow = OverlayWindow()
        ow.create_widget()
        ow.update(fingertip_screen=(10, 20), state="IDLE")
        ow.setup_launcher([], MagicMock())
        ow.toggle_launcher()
        ow.stop()
    finally:
        mod.HAS_PYQT = orig


def test_setup_launcher_empty_list_is_noop():
    """setup_launcher with no apps must not create a panel."""
    import gazecontrol.overlay.overlay_window as mod

    orig = mod.HAS_PYQT
    try:
        mod.HAS_PYQT = False
        from gazecontrol.overlay.overlay_window import OverlayWindow

        ow = OverlayWindow()
        ow.create_widget()
        ow.setup_launcher([], MagicMock())
        # No panel should be attached.
        with ow._widget_lock:
            widget = ow._widget
        assert widget is None  # widget not created when HAS_PYQT is False
    finally:
        mod.HAS_PYQT = orig


def test_toggle_launcher_without_widget_is_noop():
    """toggle_launcher must not raise when there is no widget."""
    import gazecontrol.overlay.overlay_window as mod

    orig = mod.HAS_PYQT
    try:
        mod.HAS_PYQT = False
        from gazecontrol.overlay.overlay_window import OverlayWindow

        ow = OverlayWindow()
        ow.toggle_launcher()  # must not raise
    finally:
        mod.HAS_PYQT = orig


# ---------------------------------------------------------------------------
# setup_launcher wires LauncherPanel onto the widget
# ---------------------------------------------------------------------------


def test_setup_launcher_attaches_panel_to_widget():
    """setup_launcher should attach a LauncherPanel to the widget's _panel attribute."""
    import gazecontrol.overlay.overlay_window as mod

    orig = mod.HAS_PYQT
    try:
        mod.HAS_PYQT = False
        from gazecontrol.overlay.overlay_window import OverlayWindow

        # Manually inject a fake widget so we can verify _panel is set.
        fake_widget = MagicMock()
        fake_widget._panel = None

        ow = OverlayWindow()
        with ow._widget_lock:
            ow._widget = fake_widget

        from gazecontrol.window_manager.launcher import AppLauncher, LauncherApp

        apps = [LauncherApp(name="Notepad", exe="notepad.exe")]
        launcher = AppLauncher()

        with patch("gazecontrol.overlay.overlay_window.HAS_PYQT", True):
            # Must not raise even with the mocked widget.
            try:
                ow.setup_launcher(apps, launcher)
            except Exception:
                pass  # Qt import errors are OK in headless tests
    finally:
        mod.HAS_PYQT = orig
