"""Tests for v1.0 multi-monitor/multi-user path helpers (ADR-0009)."""

from __future__ import annotations

from gazecontrol.paths import Paths


def test_gaze_profile_dir_creates_user_only(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    out = Paths.gaze_profile_dir("ciro")
    assert out == tmp_path / "ciro"
    assert out.is_dir()


def test_gaze_profile_dir_creates_user_and_monitor(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    out = Paths.gaze_profile_dir("ciro", "monitor-a")
    assert out == tmp_path / "ciro" / "monitor-a"
    assert out.is_dir()


def test_gaze_profile_v2_path_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    p = Paths.gaze_profile_v2("ciro", "monitor-a", version=3)
    assert p == tmp_path / "ciro" / "monitor-a" / "v3.npz"


def test_gaze_profile_history_sorted_ascending(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    d = Paths.gaze_profile_dir("ciro", "monitor-a")
    for n in (3, 1, 10, 2):
        (d / f"v{n}.npz").write_bytes(b"x")
    # Some noise that must be ignored.
    (d / "notes.txt").write_text("ignore me", encoding="utf-8")
    (d / "vfoo.npz").write_bytes(b"x")  # non-integer suffix

    history = Paths.gaze_profile_history("ciro", "monitor-a")

    assert [p.name for p in history] == ["v1.npz", "v2.npz", "v3.npz", "v10.npz"]


def test_gaze_profile_history_missing_dir_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    assert Paths.gaze_profile_history("nobody", "nowhere") == []


def test_legacy_gaze_profile_path_unchanged(tmp_path, monkeypatch):
    """The v1 flat path API must keep its old shape so existing installs
    can still find their profile until they run --migrate-profiles."""
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    assert Paths.gaze_profile("default") == tmp_path / "default.gaze.npz"
