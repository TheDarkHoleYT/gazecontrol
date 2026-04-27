"""Gesture feature extractor — computes normalized hand features from landmarks.

Produces a :class:`FeatureSet` dataclass instead of a raw dict, eliminating
the dead ``_thumb_tip_y`` private key that leaked into classifiers.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from gazecontrol.settings import get_settings


@dataclass
class FeatureSet:
    """Typed container for one hand's extracted features.

    All coordinates and distances are normalized to [0, 1] relative to hand/frame size.
    Velocities are in pixels/s at the configured reference resolution.
    """

    finger_states: list[int]  # [thumb, index, middle, ring, pinky]; 1=extended
    finger_angles: list[float]  # angle (degrees) of each fingertip from wrist
    palm_direction: float  # z-component of palm normal; +1=facing cam, -1=away
    hand_velocity_x: float  # horizontal centroid velocity (px/s at ref resolution)
    hand_velocity_y: float  # vertical centroid velocity
    thumb_index_distance: float  # pinch aperture, normalized by hand length
    wrist_x: float  # normalized wrist x [0, 1]
    wrist_y: float  # normalized wrist y [0, 1]
    thumb_dir_y: float  # lm[4].y - lm[2].y; positive = thumb pointing down

    def to_dict(self) -> dict[str, Any]:
        """Convert to the legacy dict format for backward-compatible classifiers."""
        return {
            "finger_states": self.finger_states,
            "finger_angles": self.finger_angles,
            "palm_direction": self.palm_direction,
            "hand_velocity_x": self.hand_velocity_x,
            "hand_velocity_y": self.hand_velocity_y,
            "thumb_index_distance": self.thumb_index_distance,
            "wrist_x": self.wrist_x,
            "wrist_y": self.wrist_y,
            "thumb_dir_y": self.thumb_dir_y,
        }

    def to_vector(self) -> list[float]:
        """Flatten to a canonical float vector for ML model inference.

        Canonical order::

            [finger_state_0..4, finger_angle_0..4, palm_direction,
             hand_velocity_x, hand_velocity_y, thumb_index_distance,
             wrist_x, wrist_y, thumb_dir_y]

        Returns:
            List of 17 floats.  Used as input to the **TCN** classifier
            (``tools/train_gesture_tcn.py``).  The MLP classifier uses a
            16-feature vector built from
            :data:`gazecontrol.gesture.mlp_classifier.FEATURE_ORDER` and does
            not currently include ``thumb_dir_y`` — see that module for the
            authoritative MLP feature order.  Retraining is required if
            either order changes.
        """
        return [
            *[float(s) for s in self.finger_states],
            *self.finger_angles,
            float(self.palm_direction),
            float(self.hand_velocity_x),
            float(self.hand_velocity_y),
            float(self.thumb_index_distance),
            float(self.wrist_x),
            float(self.wrist_y),
            float(self.thumb_dir_y),
        ]


class GestureFeatureExtractor:
    """Extract normalized gesture features from a MediaPipe hand landmark result."""

    def __init__(self, velocity_buffer_size: int = 3) -> None:
        # Buffer stores (monotonic_ts, cx, cy) so velocity is computed over
        # real wall-clock time instead of frame count.  Using frame-count as
        # denominator halves measured px/s whenever frames are dropped.
        self._centroid_buffer: deque[tuple[float, float, float]] = deque(
            maxlen=velocity_buffer_size
        )
        s = get_settings().camera
        self._ref_w = s.width
        self._ref_h = s.height

    def extract(self, hand_result: Any) -> FeatureSet | None:
        """Extract features from a hand result object.

        Returns a :class:`FeatureSet` instance, or ``None`` if no landmarks
        are present.  Classifiers that still expect a dict can call
        ``.to_dict()`` on the returned value.
        """
        if hand_result is None or not hand_result.multi_hand_landmarks:
            return None

        landmarks = hand_result.multi_hand_landmarks[0].landmark
        lm = [(lp.x, lp.y, lp.z) for lp in landmarks]

        finger_states = self._compute_finger_states(lm)
        finger_angles = self._compute_finger_angles(lm)
        palm_direction = self._compute_palm_direction(lm)
        thumb_index_distance = self._compute_thumb_index_distance(lm)
        wrist_x, wrist_y = lm[0][0], lm[0][1]
        # thumb_dir_y: positive when thumb tip is below thumb MCP (pointing down).
        thumb_dir_y = lm[4][1] - lm[2][1]

        # Centroid velocity using real wall-clock time so dropped frames don't
        # halve the measured speed.  Buffer stores (ts, cx, cy).
        pts = np.array(lm)
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        now = time.monotonic()
        self._centroid_buffer.append((now, cx, cy))

        vx, vy = 0.0, 0.0
        if len(self._centroid_buffer) >= 2:
            buf = list(self._centroid_buffer)
            dt = buf[-1][0] - buf[0][0]
            if dt > 1e-6:  # guard against zero/near-zero elapsed time
                dx = buf[-1][1] - buf[0][1]
                dy = buf[-1][2] - buf[0][2]
                vx = (dx / dt) * self._ref_w
                vy = (dy / dt) * self._ref_h

        feat = FeatureSet(
            finger_states=finger_states,
            finger_angles=finger_angles,
            palm_direction=palm_direction,
            hand_velocity_x=vx,
            hand_velocity_y=vy,
            thumb_index_distance=thumb_index_distance,
            wrist_x=wrist_x,
            wrist_y=wrist_y,
            thumb_dir_y=thumb_dir_y,
        )
        return feat

    # ------------------------------------------------------------------
    # Feature computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_finger_states(lm: list[tuple[float, float, float]]) -> list[int]:
        """Return [thumb, index, middle, ring, pinky]; 1 = extended."""
        states = [0] * 5

        # Thumb: handedness via MCP landmark positions in the flipped frame.
        is_right_hand = lm[5][0] < lm[17][0]
        states[0] = int(lm[4][0] < lm[3][0] if is_right_hand else lm[4][0] > lm[3][0])

        # Other fingers: tip.y < pip.y → extended (image y grows downward).
        for i, (tip, pip) in enumerate([(8, 6), (12, 10), (16, 14), (20, 18)]):
            states[i + 1] = int(lm[tip][1] < lm[pip][1])

        return states

    @staticmethod
    def _compute_finger_angles(lm: list[tuple[float, float, float]]) -> list[float]:
        """Angle (degrees) of each fingertip from the wrist–palm_base axis."""
        wrist = np.array(lm[0][:2])
        base_vec = np.array(lm[9][:2]) - wrist
        base_len = float(np.linalg.norm(base_vec))
        if base_len < 1e-7:
            return [0.0] * 5

        angles: list[float] = []
        for tip_idx in (4, 8, 12, 16, 20):
            tip_vec = np.array(lm[tip_idx][:2]) - wrist
            tip_len = float(np.linalg.norm(tip_vec))
            if tip_len < 1e-7:
                angles.append(0.0)
                continue
            cos_a = float(np.clip(np.dot(base_vec, tip_vec) / (base_len * tip_len), -1.0, 1.0))
            angles.append(math.degrees(math.acos(cos_a)))
        return angles

    @staticmethod
    def _compute_palm_direction(lm: list[tuple[float, float, float]]) -> float:
        """Z-component of the palm normal (+1 = facing camera)."""
        v1 = np.array(lm[5][:3]) - np.array(lm[0][:3])
        v2 = np.array(lm[17][:3]) - np.array(lm[0][:3])
        normal = np.cross(v1, v2)
        norm_len = float(np.linalg.norm(normal))
        if norm_len < 1e-7:
            return 0.0
        return float(normal[2] / norm_len)

    @staticmethod
    def _compute_thumb_index_distance(lm: list[tuple[float, float, float]]) -> float:
        """Normalized pinch aperture (thumb-tip to index-tip / hand length).

        Returns a large sentinel (1.0 = hand fully open) when hand length is
        degenerate so the PINCH rule (< 0.25) does not spuriously fire.
        """
        thumb_tip = np.array(lm[4][:2])
        index_tip = np.array(lm[8][:2])
        dist = float(np.linalg.norm(thumb_tip - index_tip))
        hand_len = float(np.linalg.norm(np.array(lm[0][:2]) - np.array(lm[9][:2])))
        if hand_len < 1e-7:
            # Degenerate hand pose — returning 0.0 would falsely trigger PINCH.
            return 1.0
        return dist / hand_len
