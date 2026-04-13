import math
import numpy as np
from collections import deque

from gazecontrol.config import FEATURE_REF_WIDTH, FEATURE_REF_HEIGHT


class GestureFeatureExtractor:
    def __init__(self, velocity_buffer_size=3):
        self._centroid_buffer = deque(maxlen=velocity_buffer_size)

    def extract(self, hand_result) -> dict | None:
        if hand_result is None or not hand_result.multi_hand_landmarks:
            return None

        landmarks = hand_result.multi_hand_landmarks[0].landmark
        lm = [(l.x, l.y, l.z) for l in landmarks]

        finger_states = self._compute_finger_states(lm)
        finger_angles = self._compute_finger_angles(lm)
        palm_direction = self._compute_palm_direction(lm)
        thumb_index_distance = self._compute_thumb_index_distance(lm)

        wrist_x, wrist_y = lm[0][0], lm[0][1]

        cx = np.mean([p[0] for p in lm])
        cy = np.mean([p[1] for p in lm])
        self._centroid_buffer.append((cx, cy))

        vx, vy = 0.0, 0.0
        if len(self._centroid_buffer) >= 2:
            buf = list(self._centroid_buffer)
            dx = buf[-1][0] - buf[0][0]
            dy = buf[-1][1] - buf[0][1]
            n = len(buf) - 1
            # BUG-9: use config resolution instead of hardcoded 640x480
            vx = (dx / n) * FEATURE_REF_WIDTH
            vy = (dy / n) * FEATURE_REF_HEIGHT

        return {
            'finger_states': finger_states,
            'finger_angles': finger_angles,
            'palm_direction': palm_direction,
            'hand_velocity_x': vx,
            'hand_velocity_y': vy,
            'thumb_index_distance': thumb_index_distance,
            'wrist_x': wrist_x,
            'wrist_y': wrist_y,
            '_thumb_tip_y': lm[4][1],
        }

    def _compute_finger_states(self, lm):
        states = [0] * 5

        # Thumb: direction dipende dalla mano.
        # Nella frame flippata: se index_MCP (lm[5]) è a sx di pinky_MCP (lm[17])
        # → mano destra. Per la mano destra il pollice è esteso quando tip.x < IP.x;
        # per la sinistra quando tip.x > IP.x.
        is_right_hand = lm[5][0] < lm[17][0]
        if is_right_hand:
            if lm[4][0] < lm[3][0]:
                states[0] = 1
        else:
            if lm[4][0] > lm[3][0]:
                states[0] = 1

        # Index, middle, ring, pinky: tip y < pip y means extended
        tip_pip = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for i, (tip, pip) in enumerate(tip_pip):
            if lm[tip][1] < lm[pip][1]:
                states[i + 1] = 1

        return states

    def _compute_finger_angles(self, lm):
        wrist = np.array(lm[0][:2])
        base_vec = np.array(lm[9][:2]) - wrist
        base_len = np.linalg.norm(base_vec)
        if base_len < 1e-7:
            return [0.0] * 5

        angles = []
        tips = [4, 8, 12, 16, 20]
        for tip_idx in tips:
            tip_vec = np.array(lm[tip_idx][:2]) - wrist
            tip_len = np.linalg.norm(tip_vec)
            if tip_len < 1e-7:
                angles.append(0.0)
                continue
            cos_a = np.clip(np.dot(base_vec, tip_vec) / (base_len * tip_len), -1, 1)
            angles.append(math.degrees(math.acos(cos_a)))
        return angles

    def _compute_palm_direction(self, lm):
        v1 = np.array(lm[5][:3]) - np.array(lm[0][:3])
        v2 = np.array(lm[17][:3]) - np.array(lm[0][:3])
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-7:
            return 0.0
        normal = normal / norm_len
        return float(normal[2])

    def _compute_thumb_index_distance(self, lm):
        thumb_tip = np.array(lm[4][:2])
        index_tip = np.array(lm[8][:2])
        dist = np.linalg.norm(thumb_tip - index_tip)
        hand_len = np.linalg.norm(np.array(lm[0][:2]) - np.array(lm[9][:2]))
        if hand_len < 1e-7:
            return 0.0
        return float(dist / hand_len)
