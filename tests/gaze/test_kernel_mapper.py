"""GazeMapper non-linear mapper modes (G10): kernel_ridge + gp.

These tests exercise the dispatch in fit() / predict() / load() for the
new v1.0 mapper types. The math is verified against synthetic data so
sklearn is the only dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from gazecontrol.gaze.gaze_mapper import GazeMapper


def _nonlinear_calibration(rng_seed: int = 0):
    """Synthetic calibration data with a deliberately non-linear warp.

    The screen is a function of yaw² and pitch² (peripheral compression
    mimicking real-world miscalibration) — poly_ridge can fit it
    perfectly because the design matrix contains those terms, but the
    test verifies that kernel_ridge and gp also recover the mapping.
    """
    rng = np.random.default_rng(rng_seed)
    angles = np.array(
        [(yaw, pitch) for yaw in range(-20, 21, 5) for pitch in range(-15, 16, 5)],
        dtype=float,
    )
    sw, sh = 1920, 1080
    targets = np.column_stack(
        [
            sw / 2
            + 30.0 * angles[:, 0]
            - 0.3 * (angles[:, 0] ** 2) * np.sign(angles[:, 0])
            + rng.normal(0, 2, len(angles)),
            sh / 2
            - 25.0 * angles[:, 1]
            + 0.4 * (angles[:, 1] ** 2) * np.sign(angles[:, 1])
            + rng.normal(0, 2, len(angles)),
        ]
    )
    return angles, targets, sw, sh


def test_ctor_rejects_unknown_mapper_type():
    with pytest.raises(ValueError, match="Unknown mapper_type"):
        GazeMapper(mapper_type="rgb")


def test_kernel_ridge_fit_then_predict_recovers_targets():
    angles, targets, sw, sh = _nonlinear_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh, mapper_type="kernel_ridge")
    mapper.fit(angles, targets)
    assert mapper.is_fitted is True
    # Reconstruct each training target within a small tolerance.
    errors = []
    for (yaw, pitch), (gx, gy) in zip(angles, targets, strict=True):
        pred = mapper.predict(yaw, pitch)
        assert pred is not None
        errors.append(np.hypot(pred[0] - gx, pred[1] - gy))
    median_err = float(np.median(errors))
    assert median_err < 80.0, f"kernel_ridge median residual too large: {median_err}"


def test_gp_predict_with_uncertainty_returns_sigma():
    angles, targets, sw, sh = _nonlinear_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh, mapper_type="gp")
    mapper.fit(angles, targets)
    result = mapper.predict_with_uncertainty(0.0, 0.0)
    assert result is not None
    (px, py), sigma = result
    assert 0 <= px < sw and 0 <= py < sh
    # GP must populate a finite, positive sigma for in-domain queries.
    assert sigma is not None and sigma > 0


def test_predict_with_uncertainty_returns_none_sigma_for_poly_ridge():
    angles, targets, sw, sh = _nonlinear_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh, mapper_type="poly_ridge")
    mapper.fit(angles, targets)
    result = mapper.predict_with_uncertainty(0.0, 0.0)
    assert result is not None
    _, sigma = result
    assert sigma is None


def test_kernel_ridge_roundtrip_via_save_load(tmp_path):
    angles, targets, sw, sh = _nonlinear_calibration()
    a = GazeMapper(screen_w=sw, screen_h=sh, mapper_type="kernel_ridge")
    a.fit(angles, targets)
    a.save(tmp_path / "kr_profile")

    b = GazeMapper(screen_w=sw, screen_h=sh)
    assert b.load(tmp_path / "kr_profile") is True
    assert b.metadata()["mapper_type"] == "kernel_ridge"
    # Predictions must match the in-memory mapper within numerical noise.
    for yaw, pitch in [(0.0, 0.0), (5.0, -3.0), (-12.0, 8.0)]:
        pa = a.predict(yaw, pitch)
        pb = b.predict(yaw, pitch)
        assert pa is not None and pb is not None
        assert pa[0] == pytest.approx(pb[0], abs=1e-3)
        assert pa[1] == pytest.approx(pb[1], abs=1e-3)


def test_gp_roundtrip_via_save_load(tmp_path):
    angles, targets, sw, sh = _nonlinear_calibration()
    a = GazeMapper(screen_w=sw, screen_h=sh, mapper_type="gp")
    a.fit(angles, targets)
    a.save(tmp_path / "gp_profile")

    b = GazeMapper(screen_w=sw, screen_h=sh)
    assert b.load(tmp_path / "gp_profile") is True
    assert b.metadata()["mapper_type"] == "gp"
    res_b = b.predict_with_uncertainty(0.0, 0.0)
    assert res_b is not None
    _, sigma_b = res_b
    assert sigma_b is not None and sigma_b > 0


def test_load_downgrades_when_kernel_profile_missing_training_data(tmp_path, caplog):
    """If a v2 npz claims mapper_type=kernel_ridge but has no inline
    training data (e.g. the file was hand-edited or a v1 migrator only
    copied coefficients), the loader must downgrade to poly_ridge
    rather than refusing the profile entirely."""
    angles, targets, sw, sh = _nonlinear_calibration()
    a = GazeMapper(screen_w=sw, screen_h=sh, mapper_type="poly_ridge")
    a.fit(angles, targets)
    a.save(tmp_path / "broken")

    # Hand-edit the meta.json to claim kernel_ridge without changing the
    # npz contents (which still has training data, so we also empty the
    # training arrays for a true downgrade simulation).
    import json

    meta_path = tmp_path / "broken.meta.json"
    npz_path = tmp_path / "broken.npz"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["mapper_type"] = "kernel_ridge"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    # Strip the training data from the npz to force the downgrade path.
    np.savez_compressed(
        npz_path,
        coef_x=a._coef_x,
        coef_y=a._coef_y,
        intercept_x=np.array([a._intercept_x]),
        intercept_y=np.array([a._intercept_y]),
        scaler_mean=a._scaler_mean,
        scaler_scale=a._scaler_scale,
        training_angles=np.array([]),
        training_targets=np.array([]),
        training_head_poses=np.array([]),
    )

    b = GazeMapper(screen_w=sw, screen_h=sh)
    with caplog.at_level("WARNING"):
        assert b.load(tmp_path / "broken") is True
    assert b.metadata()["mapper_type"] == "poly_ridge"
    assert any("no inline training data" in m for m in caplog.messages)


def test_fit_kwarg_mapper_type_overrides_ctor():
    angles, targets, _, _ = _nonlinear_calibration()
    mapper = GazeMapper(mapper_type="poly_ridge")
    mapper.fit(angles, targets, mapper_type="kernel_ridge")
    assert mapper.metadata()["mapper_type"] == "kernel_ridge"


def test_fit_rejects_unknown_mapper_type_override():
    angles, targets, _, _ = _nonlinear_calibration()
    mapper = GazeMapper()
    with pytest.raises(ValueError, match="Unknown mapper_type"):
        mapper.fit(angles, targets, mapper_type="not_a_thing")
