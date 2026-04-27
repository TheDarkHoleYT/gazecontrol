"""Rule-based gesture classifier — fast path using hand geometry heuristics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gazecontrol.settings import get_settings

if TYPE_CHECKING:
    from gazecontrol.gesture.feature_extractor import FeatureSet


class RuleClassifier:
    """Classify hand gestures using geometric rules.

    Accepts a :class:`~gazecontrol.gesture.feature_extractor.FeatureSet`
    instance or a legacy dict (for backward compatibility with test doubles).
    """

    def classify(self, features: FeatureSet | dict[str, Any] | None) -> tuple[str | None, float]:
        """Return ``(gesture_label, confidence)`` or ``(None, 0.0)``."""
        if features is None:
            return (None, 0.0)

        feat: dict[str, Any]
        feat = features.to_dict() if hasattr(features, "to_dict") else features

        s = get_settings().gesture
        swipe_thr = s.swipe_velocity_threshold

        fs = feat["finger_states"]
        vy = feat["hand_velocity_y"]

        # PINCH: thumb–index close, other fingers extended.
        if feat["thumb_index_distance"] < 0.25 and fs[2] == 1 and fs[3] == 1 and fs[4] == 1:
            return ("PINCH", 0.92)

        # SCROLL_UP / SCROLL_DOWN: index+middle extended, others closed + vertical velocity.
        if fs[1] == 1 and fs[2] == 1 and fs[0] == 0 and fs[3] == 0 and fs[4] == 0:
            if vy < -swipe_thr:
                return ("SCROLL_UP", 0.90)
            if vy > swipe_thr:
                return ("SCROLL_DOWN", 0.90)

        # RELEASE: all five fingers extended.
        if fs == [1, 1, 1, 1, 1]:
            return ("RELEASE", 0.95)

        # CLOSE_SIGN / GRAB: fist.
        if fs == [0, 0, 0, 0, 0]:
            thumb_dir_y: float = feat.get("thumb_dir_y", 0.0)
            if thumb_dir_y > 0.0:
                return ("CLOSE_SIGN", 0.90)
            return ("GRAB", 0.95)

        return (None, 0.0)
