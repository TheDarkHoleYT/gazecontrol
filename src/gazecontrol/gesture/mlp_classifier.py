"""Gesture MLP classifier — ONNX runtime inference.

Training and model export functionality has been moved to
``tools/train_gesture_mlp.py`` (not a runtime dependency).
"""
from __future__ import annotations

import logging

import numpy as np

from gazecontrol.paths import Paths

logger = logging.getLogger(__name__)

MLP_GESTURE_LABELS: list[str] = ["PINCH", "SWIPE_LEFT", "SWIPE_RIGHT", "MAXIMIZE"]


def _features_to_vector(features: dict) -> np.ndarray:
    """Flatten a feature dict to a float32 row vector."""
    vec: list[float] = []
    vec.extend(float(x) for x in features["finger_states"])
    vec.extend(float(x) for x in features["finger_angles"])
    vec.append(float(features["palm_direction"]))
    vec.append(float(features["hand_velocity_x"]))
    vec.append(float(features["hand_velocity_y"]))
    vec.append(float(features["thumb_index_distance"]))
    return np.array(vec, dtype=np.float32).reshape(1, -1)


class MLPClassifier:
    """ONNX-backed gesture MLP classifier.

    If the ONNX model is absent at construction time, the classifier starts
    in disabled state (``is_loaded() == False``) and ``classify()`` always
    returns ``(None, 0.0)``.
    """

    def __init__(self) -> None:
        self._session: object | None = None
        self._input_name: str | None = None
        self._loaded = False
        self._try_load(str(Paths.gesture_mlp_model()))

    def _try_load(self, path: str) -> None:
        import os

        if not os.path.isfile(path):
            logger.debug("Gesture MLP model not found at %s; classifier disabled.", path)
            return
        self.load(path)

    def is_loaded(self) -> bool:
        """Return True when the ONNX session is ready."""
        return self._loaded

    def load(self, path: str) -> bool:
        """Load an ONNX model from *path*.  Returns True on success."""
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(path)
            self._input_name = self._session.get_inputs()[0].name  # type: ignore[union-attr]
            self._loaded = True
            logger.info("Gesture MLP loaded from %s", path)
            return True
        except Exception as exc:
            logger.warning("Failed to load gesture MLP: %s", exc)
            self._loaded = False
            return False

    def classify(self, features: dict | None) -> tuple[str | None, float]:
        """Classify a feature dict.

        Returns:
            ``(label, confidence)`` or ``(None, 0.0)`` when classification
            is impossible (model not loaded, inference error, or bad output).
        """
        if not self._loaded or features is None or self._session is None:
            return (None, 0.0)

        vec = _features_to_vector(features)
        try:
            outputs = self._session.run(None, {self._input_name: vec})  # type: ignore[index]
            probas = outputs[1][0]
            if not isinstance(probas, dict):
                logger.warning(
                    "MLP returned non-dict probabilities (type=%s); skipping.",
                    type(probas).__name__,
                )
                return (None, 0.0)
            missing = [k for k in MLP_GESTURE_LABELS if k not in probas]
            if missing:
                logger.warning("MLP probabilities missing keys: %s", missing)
                return (None, 0.0)
            idx = int(np.argmax([probas[label] for label in MLP_GESTURE_LABELS]))
            label = MLP_GESTURE_LABELS[idx]
            return (label, float(probas[label]))
        except Exception as exc:
            logger.warning("MLP inference error: %s", exc)
            return (None, 0.0)
