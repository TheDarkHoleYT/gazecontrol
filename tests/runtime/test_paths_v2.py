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


# ---------------------------------------------------------------------------
# v2 active-profile resolution + version bump (G19)
# ---------------------------------------------------------------------------


def test_resolve_active_v2_profile_returns_none_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    assert Paths.resolve_active_v2_profile("nobody", "nowhere") is None


def test_resolve_active_v2_profile_uses_pointer_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    d = Paths.gaze_profile_dir("ciro", "monitor-a")
    (d / "v1.npz").write_bytes(b"old")
    (d / "v2.npz").write_bytes(b"new")
    (d / "latest.txt").write_text("v1\n", encoding="utf-8")
    # Pointer says v1 — must win over the newer v2.npz.
    assert Paths.resolve_active_v2_profile("ciro", "monitor-a") == d / "v1.npz"


def test_resolve_active_v2_profile_falls_back_to_highest_version(tmp_path, monkeypatch):
    """Right after the one-shot migrator there is no latest.txt yet —
    the resolver must still return the newest v{N}.npz."""
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    d = Paths.gaze_profile_dir("ciro", "monitor-a")
    (d / "v1.npz").write_bytes(b"x")
    (d / "v3.npz").write_bytes(b"x")
    assert Paths.resolve_active_v2_profile("ciro", "monitor-a") == d / "v3.npz"


def test_resolve_active_v2_profile_pointer_to_missing_file_falls_back(
    tmp_path, monkeypatch
):
    """A pointer that references a deleted version must not crash —
    the resolver falls back to the highest existing version."""
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    d = Paths.gaze_profile_dir("ciro", "monitor-a")
    (d / "v1.npz").write_bytes(b"x")
    (d / "latest.txt").write_text("v9\n", encoding="utf-8")
    assert Paths.resolve_active_v2_profile("ciro", "monitor-a") == d / "v1.npz"


def test_next_v2_profile_version_starts_at_one(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    assert Paths.next_v2_profile_version("ciro", "monitor-a") == 1


def test_next_v2_profile_version_bumps_past_max_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    d = Paths.gaze_profile_dir("ciro", "monitor-a")
    for v in (1, 3, 7):
        (d / f"v{v}.npz").write_bytes(b"x")
    assert Paths.next_v2_profile_version("ciro", "monitor-a") == 8


def test_write_latest_pointer_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: tmp_path))
    target = Paths.write_latest_pointer("ciro", "monitor-a", 5)
    assert target.read_text(encoding="utf-8").strip() == "v5"
    # The .part staging file must not survive a successful write.
    leftovers = list(target.parent.glob("*.part"))
    assert not leftovers, f"Stale .part files: {leftovers}"
