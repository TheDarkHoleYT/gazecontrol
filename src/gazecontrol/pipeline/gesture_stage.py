"""GestureStage — hand detection, feature extraction, and pointer mapping."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING

from gazecontrol.gesture.feature_extractor import GestureFeatureExtractor
from gazecontrol.gesture.pinch_tracker import PinchEvent, PinchTracker
from gazecontrol.gesture.rule_classifier import RuleClassifier
from gazecontrol.pipeline.context import FrameContext

if TYPE_CHECKING:
    from gazecontrol.gesture.fingertip_mapper import FingertipMapper
    from gazecontrol.gesture.fusion import GestureFusion
    from gazecontrol.gesture.hand_detector import HandDetector
    from gazecontrol.gesture.mlp_classifier import MLPClassifier, TCNClassifier
    from gazecontrol.settings import AppSettings, GestureSettings, InteractionSettings

logger = logging.getLogger(__name__)


class GestureStage:
    """Hand detection + gesture classification + fingertip mapping stage.

    Produces ``ctx.fingertip_screen``, ``ctx.pinch_event``,
    ``ctx.two_finger_scroll_delta``, ``ctx.gesture_label``, and
    ``ctx.gesture_confidence`` each frame.

    Heavy resources (MediaPipe HandDetector, MLP ONNX session) are NOT
    allocated in ``__init__`` — they are allocated in ``start()`` so that
    test-doubles can construct a ``GestureStage`` without loading models.
    """

    name = "gesture"

    def __init__(
        self,
        fingertip_mapper: FingertipMapper,
        settings: AppSettings | None = None,
    ) -> None:
        self._mapper = fingertip_mapper
        self._settings = settings

        # Heavy resources — populated in start(), released in stop().
        self._hand_detector: HandDetector | None = None
        self._feature_extractor: GestureFeatureExtractor | None = None
        self._rule_classifier: RuleClassifier | None = None
        self._mlp_classifier: MLPClassifier | None = None
        self._tcn_classifier: TCNClassifier | None = None
        self._fusion: GestureFusion | None = None
        self._pinch_tracker: PinchTracker | None = None

        # Timing for dt computation passed to FingertipMapper filters.
        self._last_process_ts: float = time.monotonic()

    def _get_gesture_cfg(self) -> GestureSettings:
        if self._settings is not None:
            return self._settings.gesture
        from gazecontrol.settings import get_settings

        return get_settings().gesture

    def _get_interaction_cfg(self) -> InteractionSettings:
        if self._settings is not None:
            return self._settings.interaction
        from gazecontrol.settings import get_settings

        return get_settings().interaction

    def start(self) -> bool:
        """Allocate MediaPipe, feature extractor, classifiers, and pinch tracker."""
        from gazecontrol.gesture.hand_detector import HandDetector

        try:
            self._hand_detector = HandDetector()
        except (RuntimeError, OSError, ValueError):
            logger.exception("GestureStage: failed to initialise HandDetector.")
            return False

        self._feature_extractor = GestureFeatureExtractor()
        self._rule_classifier = RuleClassifier()
        self._mlp_classifier = None
        self._tcn_classifier = None
        self._fusion = None
        self._init_mlp()
        self._init_fusion()

        icfg = self._get_interaction_cfg()
        self._pinch_tracker = PinchTracker(
            down_threshold=icfg.pinch_down_threshold,
            up_threshold=icfg.pinch_up_threshold,
        )
        return True

    def _init_mlp(self) -> None:
        try:
            from gazecontrol.gesture.mlp_classifier import MLPClassifier

            self._mlp_classifier = MLPClassifier()
        except (ImportError, RuntimeError, OSError, ValueError):
            logger.warning("MLPClassifier not loaded; using rule classifier only.", exc_info=True)

    def _init_fusion(self) -> None:
        """Set up GestureFusion with rule + TCN classifiers."""
        if self._rule_classifier is None:
            return
        try:
            from gazecontrol.gesture.fusion import GestureFusion
            from gazecontrol.gesture.mlp_classifier import TCNClassifier

            tcn = TCNClassifier()
            self._tcn_classifier = tcn
            self._fusion = GestureFusion(
                rule=self._rule_classifier,
                ml=tcn if tcn.is_loaded() else None,
            )
            logger.debug("GestureFusion initialized (TCN loaded=%s).", tcn.is_loaded())
        except (ImportError, RuntimeError, OSError, ValueError):
            logger.warning(
                "GestureFusion not initialized; falling back to rule-only.", exc_info=True
            )

    def process(self, ctx: FrameContext) -> FrameContext:
        """Detect hand landmarks, classify gesture, map fingertip, track pinch."""
        if not ctx.capture_ok or ctx.frame_rgb is None:
            return ctx
        if (
            self._hand_detector is None
            or self._feature_extractor is None
            or self._rule_classifier is None
        ):
            return ctx

        try:
            now = time.monotonic()
            dt = max(now - getattr(self, "_last_process_ts", now), 1e-6)
            self._last_process_ts = now

            ts_ms = int(ctx.t0 * 1000)
            hand_result = self._hand_detector.process(ctx.frame_rgb, ts_ms)
            ctx.hand_result = hand_result

            if hand_result is None or not hand_result.multi_hand_landmarks:
                ctx.gesture_label = None
                ctx.gesture_confidence = 0.0
                ctx.fingertip_screen = None
                # Mirror to pointer fields so HAND_ONLY mode (no fusion stage)
                # sees pointer_screen=None when the hand is lost.
                if ctx.pointer_screen is None or ctx.pointer_source == "hand":
                    ctx.pointer_screen = None
                    ctx.pointer_source = "none"
                if self._pinch_tracker is not None:
                    # No hand → treat as full open (no pinch).
                    ctx.pinch_event = PinchEvent.NONE
                ctx.two_finger_scroll_delta = 0
                # Reset pointer filters and TCN buffer when hand is lost so
                # next detection starts clean without stale velocity/state.
                self._mapper.reset()
                if self._tcn_classifier is not None:
                    self._tcn_classifier.reset()
                return ctx

            feat = self._feature_extractor.extract(hand_result)
            if feat is None:
                return ctx
            ctx.features = feat.to_dict()

            # Map index fingertip (landmark 8) to screen coordinates.
            # Pass dt and detection confidence so filters can adapt their
            # noise weighting appropriately.
            lm = hand_result.multi_hand_landmarks[0].landmark
            # Extract handedness confidence as a proxy for detection quality.
            hand_confidence = 1.0
            if hand_result.multi_handedness and hand_result.multi_handedness[0]:
                with contextlib.suppress(AttributeError, IndexError):
                    hand_confidence = float(hand_result.multi_handedness[0].classification[0].score)
            ctx.fingertip_screen = self._mapper.map(
                lm[8].x, lm[8].y, dt=dt, confidence=hand_confidence
            )
            # Default pointer_screen to fingertip; PointerFusionStage (mode B)
            # may override when present. Mode A bypasses fusion entirely.
            ctx.pointer_screen = ctx.fingertip_screen
            ctx.pointer_source = "hand"

            # Pinch tracking (FeatureSet.thumb_index_distance).
            if self._pinch_tracker is not None:
                ctx.pinch_event = self._pinch_tracker.update(feat.thumb_index_distance)

            # Two-finger scroll: index + middle extended, others closed + vertical vel.
            fs = feat.finger_states
            vy = feat.hand_velocity_y
            gcfg = self._get_gesture_cfg()
            if fs[1] == 1 and fs[2] == 1 and fs[0] == 0 and fs[3] == 0 and fs[4] == 0:
                if vy < -gcfg.swipe_velocity_threshold:
                    ctx.two_finger_scroll_delta = 120  # scroll up
                elif vy > gcfg.swipe_velocity_threshold:
                    ctx.two_finger_scroll_delta = -120  # scroll down
                else:
                    ctx.two_finger_scroll_delta = 0
            else:
                ctx.two_finger_scroll_delta = 0

            # Gesture classification — fusion (rule-first, TCN fallback) when
            # available; otherwise manual rule → MLP fallback.
            if self._fusion is not None:
                gesture_str, confidence = self._fusion.classify(feat)
            else:
                gesture_str, confidence = self._rule_classifier.classify(feat)
                if gesture_str is None and self._mlp_classifier is not None:
                    gesture_str, confidence = self._mlp_classifier.classify(feat.to_dict())

            ctx.gesture_label = gesture_str
            ctx.gesture_confidence = confidence

        except (RuntimeError, ValueError, AttributeError):
            logger.warning("GestureStage: processing failed", exc_info=True)

        return ctx

    def stop(self) -> None:
        """Release MediaPipe and ONNX resources."""
        if self._hand_detector is not None:
            try:
                self._hand_detector.close()
            except (RuntimeError, OSError):
                logger.debug("GestureStage: HandDetector.close() failed.", exc_info=True)
            self._hand_detector = None

        if self._mlp_classifier is not None:
            try:
                self._mlp_classifier.close()
            except (RuntimeError, OSError):
                logger.debug("GestureStage: MLPClassifier.close() failed.", exc_info=True)
            self._mlp_classifier = None

        if self._tcn_classifier is not None:
            try:
                self._tcn_classifier.close()
            except (RuntimeError, OSError):
                logger.debug("GestureStage: TCNClassifier.close() failed.", exc_info=True)
            self._tcn_classifier = None

        self._fusion = None

        if self._pinch_tracker is not None:
            self._pinch_tracker.reset()
            self._pinch_tracker = None

        self._feature_extractor = None
        self._rule_classifier = None
