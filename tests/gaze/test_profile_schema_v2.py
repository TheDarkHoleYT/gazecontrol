"""GazeMapper schema v2 (ADR-0009) — metadata + training data + legacy compat."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from gazecontrol.gaze.gaze_mapper import GazeMapper


def _synthetic_calibration(rng_seed: int = 42):
    rng = np.random.default_rng(rng_seed)
    angles = np.array(
        [(yaw, pitch) for yaw in range(-15, 16, 5) for pitch in range(-10, 11, 5)],
        dtype=float,
    )
    sw, sh = 1920, 1080
    targets = np.column_stack(
        [
            sw / 2 + angles[:, 0] * 30 + rng.normal(0, 4, len(angles)),
            sh / 2 - angles[:, 1] * 25 + rng.normal(0, 4, len(angles)),
        ]
    )
    return angles, targets, sw, sh


def test_fit_populates_v2_metadata():
    angles, targets, sw, sh = _synthetic_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh)

    mapper.fit(angles, targets, fit_method="9pt", holdout_error_px=42.5)

    meta = mapper.metadata()
    assert meta["schema_version"] == "2"
    assert meta["mapper_type"] == "poly_ridge"
    assert meta["fit_method"] == "9pt"
    assert meta["samples_count"] == len(angles)
    assert meta["loo_error_px"] is not None and meta["loo_error_px"] > 0
    assert meta["holdout_error_px"] == 42.5
    assert meta["calibrated_at"] is not None
    assert "yaw" in meta["feature_schema"]
    assert meta["loaded_from_legacy_v1"] is False


def test_fit_with_head_poses_extends_feature_schema():
    angles, targets, sw, sh = _synthetic_calibration()
    head_poses = np.zeros((len(angles), 3))
    mapper = GazeMapper(screen_w=sw, screen_h=sh)

    mapper.fit(angles, targets, head_poses=head_poses)

    schema = mapper.metadata()["feature_schema"]
    assert "head_yaw" in schema and "head_pitch" in schema and "head_roll" in schema


def test_save_load_roundtrip_preserves_v2_metadata(tmp_path):
    angles, targets, sw, sh = _synthetic_calibration()
    a = GazeMapper(screen_w=sw, screen_h=sh)
    a.fit(angles, targets, fit_method="13pt_holdout", holdout_error_px=33.3)
    a.set_profile_identity(user_id="ciro", monitor_id="dell-u2720q")
    a.save(tmp_path / "profile")

    b = GazeMapper(screen_w=sw, screen_h=sh)
    assert b.load(tmp_path / "profile") is True
    meta = b.metadata()
    assert meta["schema_version"] == "2"
    assert meta["fit_method"] == "13pt_holdout"
    assert meta["holdout_error_px"] == 33.3
    assert meta["user_id"] == "ciro"
    assert meta["monitor_id"] == "dell-u2720q"
    assert meta["samples_count"] == len(angles)


def test_save_persists_training_data_inline_for_partial_fit(tmp_path):
    angles, targets, sw, sh = _synthetic_calibration()
    a = GazeMapper(screen_w=sw, screen_h=sh)
    a.fit(angles, targets)
    a.save(tmp_path / "profile")

    b = GazeMapper(screen_w=sw, screen_h=sh)
    assert b.load(tmp_path / "profile") is True
    # Internal training cache must be restored byte-for-byte.
    assert b._training_angles is not None
    assert b._training_targets is not None
    np.testing.assert_allclose(b._training_angles, angles)
    np.testing.assert_allclose(b._training_targets, targets)


def test_load_v1_profile_marks_legacy_and_fills_defaults(tmp_path):
    """A v1 profile (no schema_version, no v2 metadata, no training data)
    must load with predict() still working and `loaded_from_legacy_v1=True`."""
    # Hand-craft a v1 npz + meta.json matching the v0.7/v0.8 layout.
    npz_path = tmp_path / "default.npz"
    meta_path = tmp_path / "default.meta.json"
    np.savez_compressed(
        npz_path,
        coef_x=np.array([1.0, 2.0, 0.0, 0.0, 0.0]),
        coef_y=np.array([0.0, 3.0, 0.0, 0.0, 0.0]),
        intercept_x=np.array([960.0]),
        intercept_y=np.array([540.0]),
        scaler_mean=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        scaler_scale=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    meta_path.write_text(
        json.dumps(
            {
                "format_version": "1",
                "screen_w": 1920,
                "screen_h": 1080,
                "is_fitted": True,
            }
        ),
        encoding="utf-8",
    )

    mapper = GazeMapper()
    assert mapper.load(tmp_path / "default") is True

    assert mapper.is_fitted is True
    assert mapper.loaded_from_legacy_v1 is True
    meta = mapper.metadata()
    assert meta["calibrated_at"] is None
    assert meta["loo_error_px"] is None
    assert meta["fit_method"] == "legacy_v1"
    assert meta["mapper_type"] == "poly_ridge"
    # Training data is not present in v1 → cannot partial_fit until recal.
    assert mapper._training_angles is None
    # Predict still works.
    pt = mapper.predict(0.0, 0.0)
    assert pt is not None


def test_load_unknown_schema_version_refuses(tmp_path):
    npz_path = tmp_path / "future.npz"
    meta_path = tmp_path / "future.meta.json"
    np.savez_compressed(
        npz_path,
        coef_x=np.zeros(5),
        coef_y=np.zeros(5),
        intercept_x=np.array([0.0]),
        intercept_y=np.array([0.0]),
        scaler_mean=np.zeros(5),
        scaler_scale=np.ones(5),
    )
    meta_path.write_text(
        json.dumps({"schema_version": "99", "is_fitted": True}),
        encoding="utf-8",
    )

    mapper = GazeMapper()
    assert mapper.load(tmp_path / "future") is False


def test_load_unknown_mapper_type_falls_back_to_poly_ridge(tmp_path, caplog):
    angles, targets, sw, sh = _synthetic_calibration()
    a = GazeMapper(screen_w=sw, screen_h=sh)
    a.fit(angles, targets)
    a.save(tmp_path / "profile")

    # Rewrite the meta.json with a bogus mapper_type to simulate a profile
    # created by a future, unknown extension. The loader must downgrade
    # gracefully so the user is not locked out of their calibration.
    meta_path = tmp_path / "profile.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["mapper_type"] = "nonexistent_kernel"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    b = GazeMapper(screen_w=sw, screen_h=sh)
    with caplog.at_level("WARNING"):
        assert b.load(tmp_path / "profile") is True
    assert b.metadata()["mapper_type"] == "poly_ridge"
    assert any("unknown mapper_type" in m for m in caplog.messages)


def test_fit_clears_legacy_flag():
    """After re-fitting, a previously-legacy mapper must drop the flag."""
    angles, targets, sw, sh = _synthetic_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh)
    mapper._loaded_from_legacy_v1 = True  # simulate a legacy load
    mapper.fit(angles, targets)
    assert mapper.loaded_from_legacy_v1 is False
