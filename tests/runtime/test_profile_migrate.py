"""Tests for runtime.profile_migrate (ADR-0009 one-shot migrator)."""

from __future__ import annotations

import numpy as np

from gazecontrol.runtime.profile_migrate import (
    DEFAULT_LEGACY_MONITOR,
    MigrationResult,
    migrate_profiles,
)


def _write_legacy_npz(path):
    np.savez_compressed(
        path,
        coef_x=np.zeros(5),
        coef_y=np.zeros(5),
        intercept_x=np.array([0.0]),
        intercept_y=np.array([0.0]),
        scaler_mean=np.zeros(5),
        scaler_scale=np.ones(5),
    )


def test_no_legacy_files_returns_empty(tmp_path):
    results = migrate_profiles(tmp_path)
    assert results == []


def test_migrates_single_legacy_profile(tmp_path):
    src = tmp_path / "default.gaze.npz"
    _write_legacy_npz(src)
    src_meta = tmp_path / "default.gaze.meta.json"
    src_meta.write_text('{"format_version": "1"}', encoding="utf-8")

    results = migrate_profiles(tmp_path)

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, MigrationResult)
    assert r.action == "migrated"
    assert r.source == src
    dst_dir = tmp_path / "default" / DEFAULT_LEGACY_MONITOR
    assert r.target == dst_dir / "v1.npz"
    assert (dst_dir / "v1.npz").exists()
    assert (dst_dir / "v1.meta.json").exists()
    assert (dst_dir / "latest.txt").read_text(encoding="utf-8").strip() == "v1"
    # Source must be preserved.
    assert src.exists()


def test_migrate_is_idempotent(tmp_path):
    src = tmp_path / "default.gaze.npz"
    _write_legacy_npz(src)

    first = migrate_profiles(tmp_path)
    assert len(first) == 1 and first[0].action == "migrated"
    second = migrate_profiles(tmp_path)
    assert len(second) == 1 and second[0].action == "skipped"


def test_dry_run_does_not_touch_filesystem(tmp_path):
    src = tmp_path / "default.gaze.npz"
    _write_legacy_npz(src)

    results = migrate_profiles(tmp_path, dry_run=True)

    assert len(results) == 1
    assert results[0].action == "dry_run"
    assert not (tmp_path / "default" / DEFAULT_LEGACY_MONITOR / "v1.npz").exists()


def test_migrate_with_custom_user_and_monitor(tmp_path):
    src = tmp_path / "ciro.gaze.npz"
    _write_legacy_npz(src)

    results = migrate_profiles(tmp_path, user_id="ciro", monitor_id="dell-u2720q")

    assert len(results) == 1
    dst = tmp_path / "ciro" / "dell-u2720q" / "v1.npz"
    assert results[0].target == dst
    assert dst.exists()


def test_migrate_multiple_profiles_deterministic_order(tmp_path):
    for name in ("zeta", "alpha", "mid"):
        _write_legacy_npz(tmp_path / f"{name}.gaze.npz")

    results = migrate_profiles(tmp_path)

    assert [r.source.name for r in results] == [
        "alpha.gaze.npz",
        "mid.gaze.npz",
        "zeta.gaze.npz",
    ]


def test_missing_meta_sidecar_is_ok(tmp_path):
    """Legacy profiles without a sibling meta.json must still migrate (npz only)."""
    src = tmp_path / "default.gaze.npz"
    _write_legacy_npz(src)

    results = migrate_profiles(tmp_path)

    assert results[0].action == "migrated"
    dst_dir = tmp_path / "default" / DEFAULT_LEGACY_MONITOR
    assert (dst_dir / "v1.npz").exists()
    assert not (dst_dir / "v1.meta.json").exists()
