"""GestureStage — hand detection and gesture classification."""
from __future__ import annotations

import logging

from gazecontrol.gesture.feature_extractor import GestureFeatureExtractor
from gazecontrol.gesture.hand_detector import HandDetector
from gazecontrol.gesture.rule_classifier import RuleClassifier
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.settings import get_settings

logger = logging.getLogger(__name__)


class GestureStage:
    """Hand detection + gesture classification stage.

    Tries RuleClassifier first (fast path); falls back to MLPClassifier
    (ONNX) if the rule classifier returns no decision.
    """

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._cfg = get_settings().gesture

        self._hand_detector = HandDetector()
        self._feature_extractor = GestureFeatureExtractor()
        self._rule_classifier = RuleClassifier()

        # MLP classifier — optional; loaded lazily.
        self._mlp_classifier: object | None = None
        self._init_mlp()

        # Monotonic timestamp counter for detect_for_video (ms integer).
        self._ts_ms: int = 0

    def _init_mlp(self) -> None:
        try:
            from gazecontrol.gesture.mlp_classifier import MLPClassifier

            self._mlp_classifier = MLPClassifier()
        except Exception:
            logger.warning("MLPClassifier not loaded; using rule classifier only.", exc_info=True)

    def process(self, ctx: FrameContext) -> FrameContext:
        """Detect hand landmarks and classify gesture."""
        if not ctx.capture_ok or ctx.frame_rgb is None:
            return ctx

        try:
            # Increment monotonic timestamp for MediaPipe VIDEO mode.
            self._ts_ms += 33  # approximate 30fps delta

            hand_result = self._hand_detector.process(ctx.frame_rgb, self._ts_ms)
            ctx.hand_result = hand_result

            if hand_result is None:
                ctx.gesture_label = None
                ctx.gesture_confidence = 0.0
                ctx.hand_position = None
                return ctx

            feat = self._feature_extractor.extract(hand_result)
            if feat is None:
                return ctx

            cfg = self._cfg
            ctx.hand_position = (
                feat["wrist_x"] * self._screen_w * cfg.drag_sensitivity,
                feat["wrist_y"] * self._screen_h * cfg.drag_sensitivity,
            )

            # Rule classifier (fast path).
            gesture_str, confidence = self._rule_classifier.classify(feat)

            # MLP fallback.
            if (
                gesture_str is None
                and self._mlp_classifier is not None
                and self._mlp_classifier.is_loaded()  # type: ignore[union-attr]
            ):
                gesture_str, confidence = self._mlp_classifier.classify(feat)  # type: ignore[union-attr]

            ctx.gesture_label = gesture_str
            ctx.gesture_confidence = confidence

        except Exception:
            logger.warning("GestureStage: processing failed", exc_info=True)

        return ctx

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hand_detector.close()
