"""Head-pose estimation from MediaPipe Face Landmarker keypoints (G2).

Pure helpers — the consumer (``L2CSBackend``) feeds in a dict of
``landmark_id → (x, y)`` pixel positions and gets back ``(yaw, pitch,
roll)`` in radians, with a robustness fallback when ``cv2`` or the
required landmarks are missing.

The canonical 6-point head model uses the MediaPipe Face Mesh
landmark indices documented in the Mediapipe attention mesh paper
(2019). Coordinates are in millimetres relative to the nose tip, with
the Z axis pointing into the screen, the Y axis pointing up, and the
X axis pointing to the user's right (i.e. the *camera's* left).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: MediaPipe Face Landmarker indices for the canonical 6-point set.
#: Order matters — must match :data:`_MODEL_POINTS_MM` below.
PNP_LANDMARK_IDS: tuple[int, ...] = (
    1,    # nose tip
    152,  # chin
    33,   # left eye outer corner (camera's right)
    263,  # right eye outer corner
    61,   # left mouth corner
    291,  # right mouth corner
)

#: 3D head-model points in millimetres, aligned with PNP_LANDMARK_IDS.
_MODEL_POINTS_MM: np.ndarray[Any, Any] = np.array(
    [
        [0.0, 0.0, 0.0],         # nose tip
        [0.0, -63.6, -12.5],     # chin
        [-43.3, 32.7, -26.0],    # left eye outer corner
        [43.3, 32.7, -26.0],     # right eye outer corner
        [-28.9, -28.9, -24.1],   # left mouth corner
        [28.9, -28.9, -24.1],    # right mouth corner
    ],
    dtype=np.float64,
)


def approximate_intrinsics(image_w: int, image_h: int) -> np.ndarray[Any, Any]:
    """Return a 3×3 pinhole intrinsics matrix derived from the frame size.

    Webcams ship without per-device calibration, so we approximate
    ``fx = fy = w`` (a sensible default for ~60° horizontal FOV) and
    place the principal point at the image centre. The error this
    introduces is consistent across frames and folds into the
    :class:`GazeMapper` calibration — what matters is that the head
    pose is *stable* under the same head position, not that it is
    metrically accurate.
    """
    cx = image_w / 2.0
    cy = image_h / 2.0
    return np.array(
        [
            [float(image_w), 0.0, cx],
            [0.0, float(image_w), cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def solve_head_pose(
    landmarks: Mapping[int, tuple[float, float]],
    image_size: tuple[int, int],
) -> tuple[float, float, float] | None:
    """Estimate ``(yaw, pitch, roll)`` in radians from face landmarks.

    Args:
        landmarks:  Mapping ``landmark_id → (x_px, y_px)``. Must
                    contain every id in :data:`PNP_LANDMARK_IDS`;
                    returns ``None`` when any are missing.
        image_size: ``(width, height)`` of the source frame in pixels.

    Returns:
        ``(yaw, pitch, roll)`` in radians using the same handedness as
        MediaPipe: positive yaw turns the head to the *user's* right
        (camera's left), positive pitch tilts the chin down, positive
        roll tilts the head toward the user's right shoulder. Returns
        ``None`` if ``cv2`` is unavailable, any required landmark is
        missing, or PnP fails to converge.
    """
    if image_size[0] <= 0 or image_size[1] <= 0:
        return None
    try:
        image_points = np.array(
            [landmarks[idx] for idx in PNP_LANDMARK_IDS],
            dtype=np.float64,
        )
    except KeyError:
        return None
    try:
        import cv2
    except ImportError:
        logger.debug("solve_head_pose: cv2 unavailable; head pose not estimated.")
        return None

    intrinsics = approximate_intrinsics(image_size[0], image_size[1])
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # webcams: assume no distortion
    success, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS_MM,
        image_points,
        intrinsics,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    return _rotation_matrix_to_euler(rmat)


def _rotation_matrix_to_euler(r: np.ndarray[Any, Any]) -> tuple[float, float, float]:
    """Convert a 3×3 rotation matrix to ``(yaw, pitch, roll)`` in radians.

    Decomposition follows the head-pose convention used by MediaPipe
    and the gaze-research literature:

    * **yaw**   rotation about the vertical (Y) axis — turning the
      head left/right.
    * **pitch** rotation about the horizontal (X) axis — nodding.
    * **roll**  rotation about the optical/Z axis — tilting toward a
      shoulder.

    Assuming a rotation order ``R = R_y(yaw) · R_x(pitch) · R_z(roll)``,
    the closed-form extraction is::

        yaw   = atan2( r[0,2],  r[2,2])
        pitch = atan2(-r[1,2],  sqrt(r[0,2]² + r[2,2]²))
        roll  = atan2( r[1,0],  r[1,1])

    Singular cases (``cos(pitch) ≈ 0``) fall back to a pitch-only
    solution so the consumer never sees NaN.
    """
    cos_pitch = math.sqrt(r[0, 2] * r[0, 2] + r[2, 2] * r[2, 2])
    if cos_pitch < 1e-6:
        # Gimbal lock — recover what we can; roll is undefined.
        yaw = 0.0
        pitch = math.atan2(-r[1, 2], cos_pitch)
        roll = math.atan2(-r[0, 1], r[0, 0])
    else:
        yaw = math.atan2(r[0, 2], r[2, 2])
        pitch = math.atan2(-r[1, 2], cos_pitch)
        roll = math.atan2(r[1, 0], r[1, 1])
    return float(yaw), float(pitch), float(roll)
