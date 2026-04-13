"""Rule-based gesture classifier — fast path using hand geometry heuristics."""
from __future__ import annotations

from gazecontrol.settings import get_settings


class RuleClassifier:
    """Classify hand gestures from a feature dict using geometric rules."""

    def classify(self, features: dict | None) -> tuple[str | None, float]:
        """Return ``(gesture_label, confidence)`` or ``(None, 0.0)``."""
        if features is None:
            return (None, 0.0)

        s = get_settings().gesture
        swipe_thr = s.swipe_velocity_threshold

        fs = features["finger_states"]
        vy = features["hand_velocity_y"]
        wrist_y = features["wrist_y"]

        # PINCH: thumb–index close, other fingers extended.
        if (
            features["thumb_index_distance"] < 0.25
            and fs[2] == 1
            and fs[3] == 1
            and fs[4] == 1
        ):
            return ("PINCH", 0.92)

        # SCROLL_UP / SCROLL_DOWN: index+middle extended, others closed + vertical velocity.
        if fs[1] == 1 and fs[2] == 1 and fs[0] == 0 and fs[3] == 0 and fs[4] == 0:
            if vy < -(swipe_thr / 100):
                return ("SCROLL_UP", 0.90)
            if vy > swipe_thr / 100:
                return ("SCROLL_DOWN", 0.90)

        # RELEASE: all five fingers extended.
        if fs == [1, 1, 1, 1, 1]:
            return ("RELEASE", 0.95)

        # CLOSE_SIGN / GRAB: fist.
        if fs == [0, 0, 0, 0, 0]:
            # Use wrist_y as thumb-direction proxy via finger angles.
            angles = features.get("finger_angles", [])
            thumb_angle = angles[0] if angles else 0.0
            # Thumb pointing down → larger angle from palm axis when fist is closed.
            if thumb_angle > 90 and wrist_y > 0.3:
                return ("CLOSE_SIGN", 0.90)
            return ("GRAB", 0.95)

        return (None, 0.0)
