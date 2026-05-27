"""Tests for the calibration grid helpers (G8b)."""

from __future__ import annotations

import pytest

from gazecontrol.calibration.grid import (
    ALLOWED_SUBSET_SIZES,
    BASE_INDICES,
    FULL_GRID,
    HOLDOUT_INDICES,
    split_train_holdout,
    subset_targets,
)

# ---------------------------------------------------------------------------
# FULL_GRID / index sets
# ---------------------------------------------------------------------------


def test_full_grid_has_13_points():
    assert len(FULL_GRID) == 13


def test_base_and_holdout_partition_full_grid():
    # Together they cover every index, with no overlap.
    assert set(BASE_INDICES) | set(HOLDOUT_INDICES) == set(range(13))
    assert set(BASE_INDICES) & set(HOLDOUT_INDICES) == set()


def test_base_indices_are_the_first_nine():
    assert tuple(range(9)) == BASE_INDICES


def test_holdout_indices_are_distinct_from_base_targets():
    """Holdout targets must not coincide with any base target so the
    holdout error is a real generalisation metric."""
    base_targets = {FULL_GRID[i] for i in BASE_INDICES}
    holdout_targets = {FULL_GRID[i] for i in HOLDOUT_INDICES}
    assert base_targets.isdisjoint(holdout_targets)


def test_all_grid_points_within_unit_square():
    for x, y in FULL_GRID:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


# ---------------------------------------------------------------------------
# subset_targets
# ---------------------------------------------------------------------------


def test_subset_targets_invalid_size_rejected():
    with pytest.raises(ValueError, match="Subset size"):
        subset_targets(7)
    with pytest.raises(ValueError):
        subset_targets(0)


def test_subset_targets_size_3_includes_centre():
    pts = subset_targets(3)
    assert len(pts) == 3
    # Centre of the 3×3 grid lives at (0.5, 0.5).
    assert (0.5, 0.5) in pts


def test_subset_targets_size_5_is_centre_and_four_corners():
    pts = subset_targets(5)
    assert len(pts) == 5
    assert (0.5, 0.5) in pts
    for corner in [(0.10, 0.10), (0.90, 0.10), (0.10, 0.90), (0.90, 0.90)]:
        assert corner in pts


def test_subset_targets_size_9_matches_legacy_grid():
    pts = subset_targets(9)
    assert len(pts) == 9
    # All entries belong to the base 9 of the full grid.
    for p in pts:
        assert p in FULL_GRID[:9]


def test_subset_targets_size_13_returns_full_grid():
    pts = subset_targets(13)
    assert pts == list(FULL_GRID)


def test_allowed_subset_sizes_is_exactly_documented_set():
    assert {3, 5, 9, 13} == ALLOWED_SUBSET_SIZES


# ---------------------------------------------------------------------------
# split_train_holdout
# ---------------------------------------------------------------------------


def test_split_train_holdout_partitions_frames():
    # Three frames per target for a 5-target capture: 0, 4, 9, 10, 12.
    # Targets 9, 10, 12 are in HOLDOUT_INDICES (9..12 range), so the
    # last 9 frames go to holdout.
    target_per_frame = [0] * 3 + [4] * 3 + [9] * 3 + [10] * 3 + [12] * 3
    train, holdout = split_train_holdout(target_per_frame)
    assert len(train) == 6
    assert len(holdout) == 9
    assert train == [0, 1, 2, 3, 4, 5]
    assert holdout == [6, 7, 8, 9, 10, 11, 12, 13, 14]


def test_split_train_holdout_empty_input_returns_empty_lists():
    train, holdout = split_train_holdout([])
    assert train == []
    assert holdout == []


def test_split_train_holdout_custom_holdout_indices():
    # Pretend "target 0" is held out — useful for unit-testing the helper
    # in isolation.
    train, holdout = split_train_holdout([0, 0, 1, 2, 1], holdout_indices=(0,))
    assert train == [2, 3, 4]
    assert holdout == [0, 1]


def test_split_train_holdout_all_train_when_only_base_targets():
    target_per_frame = [0, 1, 2, 3, 4, 5, 6, 7, 8] * 2
    train, holdout = split_train_holdout(target_per_frame)
    assert len(train) == 18
    assert holdout == []
