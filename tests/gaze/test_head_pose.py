"""Tests for the head-pose PnP helper (G2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("cv2")

from gazecontrol.gaze.head_pose import (
    _MODEL_POINTS_MM,
    PNP_LANDMARK_IDS,
    _rotation_matrix_to_euler,
    approximate_intrinsics,
    solve_head_pose,
)


def _project(rvec, tvec, frame_size):
    """Project the canonical 3D head model onto a synthetic frame.

    Returns a dict ``{landmark_id → (x_px, y_px)}`` matching the
    ``solve_head_pose`` input contract.
    """
    import cv2

    intrinsics = approximate_intrinsics(frame_size[0], frame_size[1])
    dist = np.zeros((4, 1), dtype=np.float64)
    pts2d, _ = cv2.projectPoints(_MODEL_POINTS_MM, rvec, tvec, intrinsics, dist)
    return {idx: tuple(pts2d[i, 0]) for i, idx in enumerate(PNP_LANDMARK_IDS)}


def test_solve_head_pose_recovers_zero_when_facing_camera():
    """A head looking straight at the camera should yield ~0 yaw/pitch/roll."""
    import cv2

    frame_size = (640, 480)
    tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)  # 60 cm in front
    rvec = np.zeros((3, 1), dtype=np.float64)
    landmarks = _project(rvec, tvec, frame_size)

    result = solve_head_pose(landmarks, frame_size)
    assert result is not None
    yaw, pitch, roll = result
    # All three angles should be near zero for a head looking straight on.
    assert abs(yaw) < math.radians(2.0)
    assert abs(pitch) < math.radians(2.0)
    assert abs(roll) < math.radians(2.0)
    # Cleanup the unused import (keeps strict linters happy in isolation).
    _ = cv2


def test_solve_head_pose_recovers_yaw_rotation():
    """A 20° yaw rotation around the Y axis should be recovered to within
    a couple of degrees by the iterative PnP solver."""
    import cv2

    frame_size = (640, 480)
    yaw_truth = math.radians(20.0)
    rvec = np.array([[0.0], [yaw_truth], [0.0]], dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)
    landmarks = _project(rvec, tvec, frame_size)

    result = solve_head_pose(landmarks, frame_size)
    assert result is not None
    yaw, _, _ = result
    assert abs(yaw - yaw_truth) < math.radians(3.0)
    _ = cv2


def test_solve_head_pose_returns_none_on_missing_landmark():
    landmarks = {idx: (10.0 + idx, 20.0 + idx) for idx in PNP_LANDMARK_IDS}
    landmarks.pop(33)  # remove one required id
    assert solve_head_pose(landmarks, (640, 480)) is None


def test_solve_head_pose_returns_none_on_zero_image_size():
    landmarks = {idx: (0.0, 0.0) for idx in PNP_LANDMARK_IDS}
    assert solve_head_pose(landmarks, (0, 480)) is None
    assert solve_head_pose(landmarks, (640, 0)) is None


def test_approximate_intrinsics_shape_and_principal_point():
    K = approximate_intrinsics(1280, 720)
    assert K.shape == (3, 3)
    assert K[0, 0] == 1280.0
    assert K[1, 1] == 1280.0
    assert K[0, 2] == 640.0
    assert K[1, 2] == 360.0
    assert K[2, 2] == 1.0


def test_rotation_matrix_to_euler_identity_returns_zeros():
    yaw, pitch, roll = _rotation_matrix_to_euler(np.eye(3))
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(0.0, abs=1e-9)
    assert roll == pytest.approx(0.0, abs=1e-9)


def test_rotation_matrix_to_euler_gimbal_lock_does_not_nan():
    """A near-singular matrix should fall back to a finite yaw, not NaN."""
    # Pure pitch of 90° → cos(pitch) = 0.
    r = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    yaw, pitch, roll = _rotation_matrix_to_euler(r)
    assert not math.isnan(yaw)
    assert not math.isnan(pitch)
    assert not math.isnan(roll)
