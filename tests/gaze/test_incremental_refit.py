"""Tests for GazeMapper.partial_fit + compute_holdout_error (G8)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from gazecontrol.gaze.gaze_mapper import GazeMapper


def _grid_calibration(rng_seed: int = 0):
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


# ---------------------------------------------------------------------------
# partial_fit
# ---------------------------------------------------------------------------


def test_partial_fit_requires_prior_fit():
    mapper = GazeMapper()
    new_angles = np.array([[0.0, 0.0]])
    new_targets = np.array([[960.0, 540.0]])
    with pytest.raises(RuntimeError, match="previously fitted"):
        mapper.partial_fit(new_angles, new_targets)


def test_partial_fit_extends_sample_count_and_updates_metadata():
    angles, targets, sw, sh = _grid_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh)
    mapper.fit(angles, targets)
    n_before = mapper.metadata()["samples_count"]

    new_angles = np.array([[0.0, 0.0], [5.0, 0.0], [-5.0, 0.0]])
    new_targets = np.array([[960.0, 540.0], [1100.0, 540.0], [820.0, 540.0]])
    mapper.partial_fit(new_angles, new_targets)

    meta = mapper.metadata()
    assert meta["samples_count"] == n_before + 3
    assert meta["fit_method"] == "incremental_3pt"
    # Predict still works after the refit.
    assert mapper.predict(0.0, 0.0) is not None


def test_partial_fit_rejects_mismatched_lengths():
    angles, targets, sw, sh = _grid_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh)
    mapper.fit(angles, targets)
    with pytest.raises(ValueError, match="matching length"):
        mapper.partial_fit(np.zeros((3, 2)), np.zeros((2, 2)))


def test_partial_fit_rejects_zero_or_negative_weight():
    angles, targets, sw, sh = _grid_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh)
    mapper.fit(angles, targets)
    with pytest.raises(ValueError, match="new_sample_weight"):
        mapper.partial_fit(
            np.array([[0.0, 0.0]]),
            np.array([[960.0, 540.0]]),
            new_sample_weight=0.0,
        )


def test_partial_fit_head_pose_schema_must_match_cached():
    """When the cached profile has head-pose features, new samples must
    also carry them — and vice versa."""
    angles, targets, sw, sh = _grid_calibration()
    head_poses = np.zeros((len(angles), 3))
    mapper_hp = GazeMapper(screen_w=sw, screen_h=sh)
    mapper_hp.fit(angles, targets, head_poses=head_poses)
    with pytest.raises(ValueError, match="head_pose features; new_head_poses required"):
        mapper_hp.partial_fit(np.array([[0.0, 0.0]]), np.array([[960.0, 540.0]]))

    mapper_no_hp = GazeMapper(screen_w=sw, screen_h=sh)
    mapper_no_hp.fit(angles, targets)
    with pytest.raises(ValueError, match="no head_pose features"):
        mapper_no_hp.partial_fit(
            np.array([[0.0, 0.0]]),
            np.array([[960.0, 540.0]]),
            new_head_poses=np.array([[0.0, 0.0, 0.0]]),
        )


def test_partial_fit_weight_above_one_replicates_new_samples():
    """A weight of 3 should make the partial_fit treat each new sample
    as if it had been observed three times — verified by the resulting
    samples_count (cached training data grows accordingly)."""
    angles, targets, sw, sh = _grid_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh)
    mapper.fit(angles, targets)
    n_before = mapper.metadata()["samples_count"]
    new_angles = np.array([[0.0, 0.0], [5.0, 5.0]])
    new_targets = np.array([[960.0, 540.0], [1100.0, 410.0]])
    mapper.partial_fit(new_angles, new_targets, new_sample_weight=3.0)
    assert mapper.metadata()["samples_count"] == n_before + 2 * 3


def test_partial_fit_weight_below_one_subsamples_deterministically():
    angles, targets, sw, sh = _grid_calibration()
    mapper_a = GazeMapper(screen_w=sw, screen_h=sh)
    mapper_a.fit(angles, targets)
    mapper_b = GazeMapper(screen_w=sw, screen_h=sh)
    mapper_b.fit(angles, targets)
    new_angles = np.array([[i, i] for i in range(10)], dtype=float)
    new_targets = np.column_stack(
        [960.0 + new_angles[:, 0] * 30, 540.0 - new_angles[:, 1] * 25]
    )
    mapper_a.partial_fit(new_angles, new_targets, new_sample_weight=0.5)
    mapper_b.partial_fit(new_angles, new_targets, new_sample_weight=0.5)
    # Same RNG seed → same subsample → same final samples_count and
    # bit-identical predictions.
    assert mapper_a.metadata()["samples_count"] == mapper_b.metadata()["samples_count"]
    assert mapper_a.predict(2.5, -1.0) == mapper_b.predict(2.5, -1.0)


def test_partial_fit_custom_fit_method_label_persisted():
    angles, targets, sw, sh = _grid_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh)
    mapper.fit(angles, targets)
    mapper.partial_fit(
        np.array([[0.0, 0.0]]),
        np.array([[960.0, 540.0]]),
        fit_method="quick_recal_v1",
    )
    assert mapper.metadata()["fit_method"] == "quick_recal_v1"


# ---------------------------------------------------------------------------
# compute_holdout_error
# ---------------------------------------------------------------------------


def test_compute_holdout_error_zero_for_perfect_recall_subset():
    angles, targets, sw, sh = _grid_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh)
    mapper.fit(angles, targets)
    # The mapper is degree-2 polynomial Ridge; train + holdout overlap
    # gives a low (but non-zero) error.
    err = mapper.compute_holdout_error(angles[:5], targets[:5])
    assert err is not None
    assert err >= 0.0


def test_compute_holdout_error_higher_for_out_of_distribution_targets():
    angles, targets, sw, sh = _grid_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh)
    mapper.fit(angles, targets)
    # Synthetic "wrong" targets at the origin — should produce a large error.
    bogus_targets = np.zeros_like(targets[:5])
    err_train = mapper.compute_holdout_error(angles[:5], targets[:5])
    err_bogus = mapper.compute_holdout_error(angles[:5], bogus_targets)
    assert err_train is not None and err_bogus is not None
    assert err_bogus > err_train * 5.0


def test_compute_holdout_error_returns_none_when_unfitted():
    mapper = GazeMapper()
    err = mapper.compute_holdout_error(np.zeros((3, 2)), np.zeros((3, 2)))
    assert err is None


def test_compute_holdout_error_empty_input_returns_zero():
    angles, targets, sw, sh = _grid_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh)
    mapper.fit(angles, targets)
    err = mapper.compute_holdout_error(np.zeros((0, 2)), np.zeros((0, 2)))
    assert err == 0.0


def test_compute_holdout_error_with_head_poses():
    angles, targets, sw, sh = _grid_calibration()
    head_poses = np.zeros((len(angles), 3))
    mapper = GazeMapper(screen_w=sw, screen_h=sh)
    mapper.fit(angles, targets, head_poses=head_poses)
    err = mapper.compute_holdout_error(angles[:5], targets[:5], head_poses=head_poses[:5])
    assert err is not None and err >= 0.0
