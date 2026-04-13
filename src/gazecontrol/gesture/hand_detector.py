"""HandDetector — MediaPipe HandLandmarker (Tasks API, VIDEO mode).

Uses ``detect_for_video(image, ts_ms)`` with a monotonic timestamp so that
MediaPipe can apply inter-frame tracking smoothing correctly.
"""
from __future__ import annotations

import logging
import time

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from gazecontrol.paths import Paths
from gazecontrol.settings import get_settings
from gazecontrol.utils.model_downloader import ensure_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibility wrapper: emulate legacy mp.solutions.hands result format
# so that GestureFeatureExtractor.extract() requires no changes.
# ---------------------------------------------------------------------------


class _Landmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, lm: object) -> None:
        self.x: float = lm.x  # type: ignore[attr-defined]
        self.y: float = lm.y  # type: ignore[attr-defined]
        self.z: float = lm.z  # type: ignore[attr-defined]


class _HandLandmarks:
    def __init__(self, landmarks: list) -> None:
        self.landmark = [_Landmark(lm) for lm in landmarks]


class _HandResult:
    """Emulates ``mp.solutions.hands.Hands.process()`` result."""

    def __init__(self, hand_landmarks_list: list) -> None:
        self.multi_hand_landmarks = [_HandLandmarks(lms) for lms in hand_landmarks_list]
        self.multi_handedness = [None] * len(hand_landmarks_list)


# ---------------------------------------------------------------------------


class HandDetector:
    """Detect 21 hand landmarks per hand with MediaPipe Tasks API (VIDEO mode)."""

    def __init__(self) -> None:
        s = get_settings().gesture
        model_path = ensure_model("hand_landmarker.task", str(Paths.models()))

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,  # enables detect_for_video
            num_hands=s.max_hands,
            min_hand_detection_confidence=s.min_detection_confidence,
            min_hand_presence_confidence=s.min_detection_confidence,
            min_tracking_confidence=s.min_tracking_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._start_time_ms: int = int(time.monotonic() * 1000)
        logger.info("HandDetector (Tasks API, VIDEO mode) initialized.")

    def process(
        self,
        frame_rgb: object,  # np.ndarray RGB uint8
        ts_ms: int | None = None,
    ) -> _HandResult | None:
        """Detect hand landmarks in *frame_rgb*.

        Args:
            frame_rgb: RGB frame as numpy array (H, W, 3) uint8.
            ts_ms:     Monotonic timestamp in milliseconds.  If None,
                       the current monotonic time is used.

        Returns:
            Compatible hand result object, or None if no hand detected.
        """
        try:
            if ts_ms is None:
                ts_ms = int(time.monotonic() * 1000) - self._start_time_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)  # type: ignore[arg-type]
            result = self._detector.detect_for_video(mp_image, ts_ms)

            if not result.hand_landmarks:
                return None

            return _HandResult(result.hand_landmarks)

        except Exception:
            logger.exception("HandDetector.process() failed.")
            return None

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self._detector.close()
            logger.info("HandDetector closed.")
        except Exception:
            logger.exception("HandDetector.close() failed.")
