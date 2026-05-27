"""Tests for the monitor-id helper (G21)."""

from __future__ import annotations

import re

from gazecontrol.gaze.monitor_id import (
    DEFAULT_MONITOR_ID,
    detect_active_monitor_id,
    monitor_id_from_screen_info,
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,16}__[0-9a-f]{12}$")


def test_id_format_matches_pattern():
    mid = monitor_id_from_screen_info("HDMI-1", (0, 0, 1920, 1080))
    assert _ID_PATTERN.match(mid), f"unexpected id format: {mid!r}"


def test_id_deterministic_for_same_inputs():
    a = monitor_id_from_screen_info("HDMI-1", (0, 0, 1920, 1080))
    b = monitor_id_from_screen_info("HDMI-1", (0, 0, 1920, 1080))
    assert a == b


def test_id_changes_when_geometry_changes():
    a = monitor_id_from_screen_info("HDMI-1", (0, 0, 1920, 1080))
    b = monitor_id_from_screen_info("HDMI-1", (0, 0, 3840, 2160))
    assert a != b


def test_id_changes_when_name_changes():
    a = monitor_id_from_screen_info("HDMI-1", (0, 0, 1920, 1080))
    b = monitor_id_from_screen_info("DP-2", (0, 0, 1920, 1080))
    assert a != b


def test_id_safe_against_windows_style_names():
    """Windows screen names contain backslashes and dots that must not
    leak into the on-disk directory name."""
    mid = monitor_id_from_screen_info("\\\\.\\DISPLAY1", (0, 0, 1920, 1080))
    assert "\\" not in mid
    assert "." not in mid
    assert _ID_PATTERN.match(mid)


def test_id_strips_leading_trailing_punctuation_from_prefix():
    mid = monitor_id_from_screen_info("...HDMI-1...", (0, 0, 1, 1))
    # The prefix segment must not start with '-' or '_'.
    prefix = mid.split("__", 1)[0]
    assert not prefix.startswith(("-", "_"))


def test_id_falls_back_to_screen_on_empty_name():
    mid = monitor_id_from_screen_info("", (0, 0, 1, 1))
    assert mid.startswith("screen__")


def test_id_does_not_collide_with_default_legacy_sentinel():
    """The DEFAULT_MONITOR_ID ("primary-legacy") sentinel must never be
    produced organically — the migrator owns that bucket."""
    samples = [
        ("HDMI-1", (0, 0, 1920, 1080)),
        ("DP-2", (0, 0, 3840, 2160)),
        ("\\\\.\\DISPLAY1", (0, 0, 2560, 1440)),
        ("primary-legacy", (0, 0, 100, 100)),
        ("", (0, 0, 0, 0)),
    ]
    for name, geo in samples:
        assert monitor_id_from_screen_info(name, geo) != DEFAULT_MONITOR_ID


def test_detect_active_monitor_returns_default_without_qt(monkeypatch):
    """When Qt is not importable the helper must not raise."""
    import sys

    # Hide PyQt6 from the importer for this call.
    monkeypatch.setitem(sys.modules, "PyQt6.QtGui", None)
    assert detect_active_monitor_id() == DEFAULT_MONITOR_ID


def test_detect_active_monitor_custom_default(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "PyQt6.QtGui", None)
    assert detect_active_monitor_id(default="ci") == "ci"
