"""
HandDetector - Wrapper MediaPipe HandLandmarker (Tasks API, mediapipe >= 0.10).
Restituisce un oggetto compatibile con il vecchio formato multi_hand_landmarks.
"""
import logging
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import gazecontrol.config as config
from gazecontrol.utils.model_downloader import ensure_model

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')


# ---------------------------------------------------------------------------
# Wrapper di compatibilità: emula il vecchio risultato mp.solutions.hands
# così feature_extractor.py non richiede modifiche.
# ---------------------------------------------------------------------------

class _Landmark:
    __slots__ = ('x', 'y', 'z')

    def __init__(self, lm):
        self.x = lm.x
        self.y = lm.y
        self.z = lm.z


class _HandLandmarks:
    def __init__(self, landmarks):
        self.landmark = [_Landmark(lm) for lm in landmarks]


class _HandResult:
    """Emula il risultato di mp.solutions.hands.Hands.process()."""

    def __init__(self, hand_landmarks_list):
        self.multi_hand_landmarks = [_HandLandmarks(lms) for lms in hand_landmarks_list]
        # multi_handedness non usato da feature_extractor, ma manteniamo la struttura
        self.multi_handedness = [None] * len(hand_landmarks_list)


# ---------------------------------------------------------------------------

class HandDetector:
    """Rileva 21 landmark per mano con MediaPipe Tasks API."""

    def __init__(self):
        model_path = ensure_model('hand_landmarker.task', MODELS_DIR)

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=config.HAND_MAX_HANDS,
            min_hand_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        logger.info("HandDetector (Tasks API) inizializzato")

    def process(self, frame_rgb):
        """
        Processa un frame RGB e ritorna un oggetto compatibile con il vecchio
        risultato Hands (.multi_hand_landmarks[0].landmark[i].x/y/z).
        Ritorna None se nessuna mano rilevata.
        """
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._detector.detect(mp_image)

            if not result.hand_landmarks:
                return None

            return _HandResult(result.hand_landmarks)

        except Exception:
            logger.exception("Errore durante process HandDetector")
            return None

    def close(self):
        try:
            self._detector.close()
            logger.info("HandDetector chiuso")
        except Exception:
            logger.exception("Errore durante chiusura HandDetector")
