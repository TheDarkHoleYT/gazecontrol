"""Calibration grid topology + holdout split + subset selection (G8b).

Pure helpers used by :mod:`calibration.runner` so the grid math is
unit-testable without Qt / cv2.

Topology
--------
The full v1.0 grid is **13 points** — the legacy 3×3 corners + centre
plus four "validation" points that sit at the midpoints of each edge
quadrant (between the corners and the edge centres). Train on the 9
base points, hold the 4 validation points out, report a real
holdout error alongside the LOO metric (the LOO error on a degree-2
poly ridge is optimistic).

Subset selection
----------------
For incremental "top-up" recalibration (``--calibrate-incremental N``)
the runner only shows ``N`` targets, picked deterministically:

* ``N=3`` — top-left corner, centre, bottom-right corner.
* ``N=5`` — centre + four corners.
* ``N=9`` — the legacy 3×3 grid (no holdout).
* ``N=13`` — full grid (9 train + 4 holdout). Default for first-time
  calibration.

Other values are rejected so the CLI surface stays predictable.
"""

from __future__ import annotations

from collections.abc import Iterable

#: 3×3 base grid (normalised, 0..1). Order: row-major, top-left first.
_BASE_9: tuple[tuple[float, float], ...] = (
    (0.10, 0.10), (0.50, 0.10), (0.90, 0.10),
    (0.10, 0.50), (0.50, 0.50), (0.90, 0.50),
    (0.10, 0.90), (0.50, 0.90), (0.90, 0.90),
)

#: Four randomised holdout targets at the edge-quadrant midpoints. These
#: deliberately do NOT coincide with the base 9 so the holdout error is
#: a real generalisation metric.
_HOLDOUT_4: tuple[tuple[float, float], ...] = (
    (0.30, 0.30),
    (0.70, 0.30),
    (0.30, 0.70),
    (0.70, 0.70),
)

#: Full 13-point grid: 9 base + 4 holdout. ``FULL_GRID[i]`` is in
#: ``HOLDOUT_INDICES`` iff ``i >= 9``.
FULL_GRID: tuple[tuple[float, float], ...] = _BASE_9 + _HOLDOUT_4

#: Indices (into FULL_GRID) of the 9 training points.
BASE_INDICES: tuple[int, ...] = tuple(range(9))

#: Indices (into FULL_GRID) of the 4 holdout points.
HOLDOUT_INDICES: tuple[int, ...] = tuple(range(9, 13))

#: Allowed subset sizes for :func:`subset_targets`. Includes 13 so the
#: default calibration flow can call the same helper.
ALLOWED_SUBSET_SIZES: frozenset[int] = frozenset({3, 5, 9, 13})


def subset_targets(n: int) -> list[tuple[float, float]]:
    """Return the normalised targets for an ``--calibrate-incremental N`` run.

    Args:
        n: One of 3 / 5 / 9 / 13. Other values raise ``ValueError``.

    Returns:
        ``n`` ``(x_norm, y_norm)`` tuples drawn from :data:`FULL_GRID`.
    """
    if n not in ALLOWED_SUBSET_SIZES:
        raise ValueError(
            f"Subset size {n!r} not in {sorted(ALLOWED_SUBSET_SIZES)}"
        )
    if n == 13:
        return list(FULL_GRID)
    if n == 9:
        return list(_BASE_9)
    if n == 5:
        # Centre + four corners.
        return [_BASE_9[4], _BASE_9[0], _BASE_9[2], _BASE_9[6], _BASE_9[8]]
    # n == 3 — diagonal anchor + centre.
    return [_BASE_9[0], _BASE_9[4], _BASE_9[8]]


def split_train_holdout(
    captured_target_indices: Iterable[int],
    *,
    holdout_indices: tuple[int, ...] = HOLDOUT_INDICES,
) -> tuple[list[int], list[int]]:
    """Partition captured frames into train / holdout buckets.

    Each calibration frame is tagged with the index (into ``FULL_GRID``)
    of the target it was fixating. The runner concatenates those tags
    into a flat list (one entry per captured frame); this helper
    returns ``(train_frame_indices, holdout_frame_indices)`` — *frame*
    indices, i.e. positions into the flat captured arrays.

    Example::

        # captured 5 frames for target 0, 5 frames for target 4, …
        target_index_per_frame = [0]*5 + [4]*5 + [10]*5
        train, holdout = split_train_holdout(target_index_per_frame)
        # train = [0..9]  (target 0 + target 4)
        # holdout = [10..14]  (target 10 is in HOLDOUT_INDICES)
    """
    holdout_set = set(holdout_indices)
    train: list[int] = []
    held: list[int] = []
    for frame_idx, target_idx in enumerate(captured_target_indices):
        if target_idx in holdout_set:
            held.append(frame_idx)
        else:
            train.append(frame_idx)
    return train, held
