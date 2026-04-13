"""Tests for GazeMapper — fit, predict, save/load round-trip."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gazecontrol.gaze.gaze_mapper import GazeMapper


@pytest.fixture
def calibrated_mapper() -> GazeMapper:
    """GazeMapper fitted on a synthetic 5×5 grid."""
    mapper = GazeMapper(screen_w=1920, screen_h=1080)
    rng = np.random.default_rng(42)

    # Synthetic calibration: yaw ∈ [-30, 30], pitch ∈ [-20, 20] → screen grid.
    n = 25
    yaw = rng.uniform(-30, 30, n)
    pitch = rng.uniform(-20, 20, n)
    gaze_angles = np.column_stack([yaw, pitch])

    # Linear ground truth + small noise.
    px_x = (yaw / 30.0 + 1) / 2 * 1920 + rng.normal(0, 5, n)
    px_y = (pitch / 20.0 + 1) / 2 * 1080 + rng.normal(0, 5, n)
    screen_points = np.column_stack([px_x, px_y])

    mapper.fit(gaze_angles, screen_points)
    return mapper


def test_unfitted_predict_returns_none():
    mapper = GazeMapper()
    assert mapper.predict(0.0, 0.0) is None
    assert mapper.is_fitted is False


def test_fitted_predict_returns_tuple(calibrated_mapper):
    result = calibrated_mapper.predict(0.0, 0.0)
    assert result is not None
    px_x, px_y = result
    assert 0.0 <= px_x <= 1920
    assert 0.0 <= px_y <= 1080


def test_fit_returns_loo_error(calibrated_mapper):
    # LOO error should be finite and positive.
    rng = np.random.default_rng(1)
    n = 10
    gaze_angles = np.column_stack([rng.uniform(-10, 10, n), rng.uniform(-10, 10, n)])
    screen_points = np.column_stack([rng.uniform(0, 1920, n), rng.uniform(0, 1080, n)])
    mapper = GazeMapper()
    loo = mapper.fit(gaze_angles, screen_points)
    assert np.isfinite(loo)
    assert loo >= 0.0


def test_predict_clamps_to_screen(calibrated_mapper):
    # Extreme angles should still produce in-range coords.
    result = calibrated_mapper.predict(90.0, 90.0)
    assert result is not None
    px_x, px_y = result
    assert 0.0 <= px_x <= 1919
    assert 0.0 <= px_y <= 1079


def test_save_load_round_trip(calibrated_mapper, tmp_path):
    save_path = tmp_path / "mapper"
    calibrated_mapper.save(save_path)

    assert (tmp_path / "mapper.npz").exists()
    assert (tmp_path / "mapper.meta.json").exists()

    loaded = GazeMapper()
    assert loaded.load(save_path) is True
    assert loaded.is_fitted is True

    # Predictions should match (within float precision).
    for yaw, pitch in [(-10.0, -5.0), (0.0, 0.0), (15.0, 10.0)]:
        orig = calibrated_mapper.predict(yaw, pitch)
        reloaded = loaded.predict(yaw, pitch)
        assert orig is not None
        assert reloaded is not None
        assert abs(orig[0] - reloaded[0]) < 1e-4
        assert abs(orig[1] - reloaded[1]) < 1e-4


def test_load_bad_file_returns_false(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a valid npz")
    mapper = GazeMapper()
    assert mapper.load(bad) is False
    assert mapper.is_fitted is False


def test_meta_json_content(calibrated_mapper, tmp_path):
    calibrated_mapper.save(tmp_path / "mapper")
    meta = json.loads((tmp_path / "mapper.meta.json").read_text())
    assert meta["screen_w"] == 1920
    assert meta["screen_h"] == 1080
    assert meta["is_fitted"] is True
    assert meta["format_version"] == "1"
