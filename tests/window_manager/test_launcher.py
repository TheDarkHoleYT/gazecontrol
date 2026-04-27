"""Tests for AppLauncher — subprocess-based app launcher."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gazecontrol.window_manager.launcher import AppLauncher, LauncherApp


def test_launch_calls_popen():
    app = LauncherApp(name="Notepad", exe="notepad.exe")
    launcher = AppLauncher()
    with patch("subprocess.Popen") as mock_popen:
        launcher.launch(app)
    mock_popen.assert_called_once()
    args = mock_popen.call_args[0][0]
    assert "notepad.exe" in args


def test_launch_with_args():
    app = LauncherApp(name="App", exe="app.exe", args=("--flag", "value"))
    launcher = AppLauncher()
    with patch("subprocess.Popen") as mock_popen:
        launcher.launch(app)
    args = mock_popen.call_args[0][0]
    assert "--flag" in args
    assert "value" in args


def test_launch_file_not_found_logs_warning():
    app = LauncherApp(name="Ghost", exe="nonexistent_exe_xyz.exe")
    launcher = AppLauncher()
    with patch("subprocess.Popen", side_effect=FileNotFoundError):
        launcher.launch(app)  # must not raise


def test_launch_generic_exception_logs_warning():
    app = LauncherApp(name="Bad", exe="bad.exe")
    launcher = AppLauncher()
    with patch("subprocess.Popen", side_effect=PermissionError("denied")):
        launcher.launch(app)  # must not raise


def test_launcher_app_immutable():
    app = LauncherApp(name="X", exe="x.exe")
    with pytest.raises((AttributeError, TypeError)):
        app.name = "Y"  # type: ignore[misc]


def test_launcher_app_with_icon():
    app = LauncherApp(name="X", exe="x.exe", icon="icon.png")
    assert app.icon == "icon.png"
