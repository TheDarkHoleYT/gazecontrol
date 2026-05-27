"""Tests for the multi-face disambiguation helpers (G5)."""

from __future__ import annotations

import pytest

from gazecontrol.gaze.face_tracking import (
    FaceTracker,
    iou,
    score_candidate,
)

# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------


def test_iou_identical_bbox_is_one():
    assert iou((0.1, 0.1, 0.5, 0.5), (0.1, 0.1, 0.5, 0.5)) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert iou((0.0, 0.0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0)) == 0.0


def test_iou_half_overlap():
    # 0.5 width overlap, full height overlap.
    a = (0.0, 0.0, 0.4, 0.4)  # area 0.16
    b = (0.2, 0.0, 0.6, 0.4)  # area 0.16
    # Intersection: (0.2..0.4) x (0..0.4) = 0.2 * 0.4 = 0.08
    # Union: 0.16 + 0.16 - 0.08 = 0.24
    # IoU = 0.08 / 0.24 = 1/3
    assert iou(a, b) == pytest.approx(1.0 / 3.0)


def test_iou_degenerate_bbox_returns_zero():
    """Zero-area bboxes (collapsed to a point or a line) must not raise."""
    assert iou((0.5, 0.5, 0.5, 0.5), (0.0, 0.0, 1.0, 1.0)) == 0.0


def test_iou_symmetric():
    a = (0.1, 0.1, 0.6, 0.6)
    b = (0.3, 0.2, 0.5, 0.8)
    assert iou(a, b) == iou(b, a)


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------


def test_score_candidate_no_prev_uses_only_area_and_score():
    # No prev_bbox → iou contribution is zero.
    s = score_candidate((0.0, 0.0, 0.5, 0.5), score=0.9)
    # area=0.25 ; default w_area=0.4 ; score=0.9 ; w_score=0.2
    assert s == pytest.approx(0.4 * 0.25 + 0.2 * 0.9)


def test_score_candidate_prev_bbox_boosts_high_iou():
    """A bbox overlapping the last tracked face must outscore a slightly
    larger bbox at a disjoint location (sticky tracking)."""
    prev = (0.45, 0.45, 0.55, 0.55)
    sticky = (0.43, 0.43, 0.57, 0.57)  # small, overlaps prev
    bigger_far = (0.0, 0.0, 0.30, 0.30)  # larger area but no overlap
    s_sticky = score_candidate(sticky, 0.8, prev_bbox=prev)
    s_far = score_candidate(bigger_far, 0.8, prev_bbox=prev)
    assert s_sticky > s_far


def test_score_candidate_score_clamped_to_unit():
    """Detector confidences outside [0, 1] should not push the composite
    score arbitrarily — they are clamped."""
    a = score_candidate((0.0, 0.0, 0.1, 0.1), score=5.0)
    b = score_candidate((0.0, 0.0, 0.1, 0.1), score=1.0)
    assert a == pytest.approx(b)
    neg = score_candidate((0.0, 0.0, 0.1, 0.1), score=-1.0)
    zero = score_candidate((0.0, 0.0, 0.1, 0.1), score=0.0)
    assert neg == pytest.approx(zero)


# ---------------------------------------------------------------------------
# FaceTracker
# ---------------------------------------------------------------------------


def test_face_tracker_first_detection_gets_id_one():
    t = FaceTracker()
    result = t.update([((0.4, 0.4, 0.6, 0.6), 0.9)])
    assert result is not None
    bbox, face_id, multi = result
    assert bbox == (0.4, 0.4, 0.6, 0.6)
    assert face_id == 1
    assert multi is False


def test_face_tracker_sticky_when_overlap_above_threshold():
    t = FaceTracker(lock_iou_threshold=0.3)
    t.update([((0.4, 0.4, 0.6, 0.6), 0.9)])
    # Second frame: same face moved slightly.
    result = t.update([((0.42, 0.41, 0.58, 0.59), 0.9)])
    assert result is not None
    _, face_id, _ = result
    assert face_id == 1  # id reused


def test_face_tracker_rotates_id_when_iou_below_threshold():
    t = FaceTracker(lock_iou_threshold=0.5)
    t.update([((0.0, 0.0, 0.2, 0.2), 0.9)])
    # Second frame: completely different face position → new id.
    result = t.update([((0.7, 0.7, 0.9, 0.9), 0.9)])
    assert result is not None
    _, face_id, _ = result
    assert face_id == 2


def test_face_tracker_multi_flag_when_two_detections():
    t = FaceTracker()
    result = t.update(
        [
            ((0.0, 0.0, 0.2, 0.2), 0.8),  # small face on the left
            ((0.4, 0.4, 0.8, 0.8), 0.8),  # bigger face in the centre
        ]
    )
    assert result is not None
    bbox, _, multi = result
    assert multi is True
    # Larger area should win the first frame (no sticky prev).
    assert bbox == (0.4, 0.4, 0.8, 0.8)


def test_face_tracker_prefers_sticky_face_in_crowded_scene():
    """Once we have locked a face, a slightly bigger but unrelated face
    must not steal the lock unless it overlaps."""
    t = FaceTracker(lock_iou_threshold=0.3)
    t.update([((0.4, 0.4, 0.6, 0.6), 0.9)])  # area 0.04 — locked
    # Next frame: the locked face shifted slightly, a bigger newcomer appears.
    result = t.update(
        [
            ((0.42, 0.41, 0.58, 0.59), 0.9),  # sticky
            ((0.0, 0.0, 0.3, 0.3), 0.9),  # bigger, no overlap
        ]
    )
    assert result is not None
    bbox, face_id, multi = result
    assert face_id == 1
    assert bbox == (0.42, 0.41, 0.58, 0.59)
    assert multi is True


def test_face_tracker_returns_none_on_empty_detections():
    t = FaceTracker()
    assert t.update([]) is None


def test_face_tracker_reset_drops_lock():
    t = FaceTracker()
    t.update([((0.4, 0.4, 0.6, 0.6), 0.9)])
    t.reset()
    assert t.current_face_id is None
    assert t.current_bbox is None
    # New detection after reset → id increments past the previous count.
    result = t.update([((0.4, 0.4, 0.6, 0.6), 0.9)])
    assert result is not None
    _, face_id, _ = result
    assert face_id == 2  # counter does not reset, only the lock does


def test_face_tracker_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        FaceTracker(lock_iou_threshold=0.0)
    with pytest.raises(ValueError):
        FaceTracker(lock_iou_threshold=1.5)


def test_face_tracker_exposes_last_state():
    t = FaceTracker()
    assert t.current_face_id is None
    t.update([((0.0, 0.0, 0.5, 0.5), 0.7)])
    assert t.current_face_id == 1
    assert t.current_bbox == (0.0, 0.0, 0.5, 0.5)
