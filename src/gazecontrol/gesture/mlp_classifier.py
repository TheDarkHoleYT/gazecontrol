"""Gesture MLP and TCN classifiers — ONNX runtime inference.

Training and model export functionality has been moved to
``tools/train_gesture_mlp.py`` and ``tools/train_gesture_tcn.py``
(not runtime dependencies).
"""

from __future__ import annotations

import enum
import json
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gazecontrol.gesture.labels import DEFAULT_LABELS
from gazecontrol.paths import Paths

if TYPE_CHECKING:
    from gazecontrol.gesture.feature_extractor import FeatureSet

logger = logging.getLogger(__name__)

MLP_GESTURE_LABELS: list[str] = ["PINCH", "SWIPE_LEFT", "SWIPE_RIGHT", "MAXIMIZE"]

# Canonical feature order shared between inference and training.  If the
# trainer adds or reorders features, update this tuple and retrain.
#
# NB: this order intentionally OMITS ``thumb_dir_y`` (which is exposed by
# :class:`~gazecontrol.gesture.feature_extractor.FeatureSet.to_vector` and
# consumed by the TCN classifier).  The MLP shipped in ``models/`` was
# trained on 16 features; adding ``thumb_dir_y`` here without retraining the
# ONNX would cause an input-shape mismatch at inference time.
FEATURE_ORDER: tuple[str, ...] = (
    *[f"finger_state_{i}" for i in range(5)],  # 5 finger states
    *[f"finger_angle_{i}" for i in range(5)],  # 5 finger angles
    "palm_direction",
    "hand_velocity_x",
    "hand_velocity_y",
    "thumb_index_distance",
    "wrist_x",
    "wrist_y",
)


class _OutputKind(enum.Enum):
    ZIPMAP = "zipmap"  # outputs[1][0] is dict[label, prob]
    NDARRAY_PROBS = "ndarray"  # outputs[1] is (N,) float32 probability array
    UNKNOWN = "unknown"


def _features_to_vector(features: dict[str, Any]) -> np.ndarray[Any, Any]:
    """Flatten a feature dict to a float32 row vector using canonical FEATURE_ORDER."""
    # finger_states and finger_angles are stored as lists in the dict.
    finger_states = list(features.get("finger_states", [0.0] * 5))
    finger_angles = list(features.get("finger_angles", [0.0] * 5))
    expanded = {
        **{f"finger_state_{i}": float(finger_states[i]) for i in range(min(5, len(finger_states)))},
        **{f"finger_angle_{i}": float(finger_angles[i]) for i in range(min(5, len(finger_angles)))},
        "palm_direction": float(features.get("palm_direction", 0.0)),
        "hand_velocity_x": float(features.get("hand_velocity_x", 0.0)),
        "hand_velocity_y": float(features.get("hand_velocity_y", 0.0)),
        "thumb_index_distance": float(features.get("thumb_index_distance", 0.0)),
        "wrist_x": float(features.get("wrist_x", 0.0)),
        "wrist_y": float(features.get("wrist_y", 0.0)),
    }
    vec = [expanded.get(k, 0.0) for k in FEATURE_ORDER]
    return np.array(vec, dtype=np.float32).reshape(1, -1)


class MLPClassifier:
    """ONNX-backed gesture MLP classifier.

    If the ONNX model is absent at construction time, the classifier starts
    in disabled state (``is_loaded() == False``) and ``classify()`` always
    returns ``(None, 0.0)``.
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._input_name: str | None = None
        self._output_kind: _OutputKind = _OutputKind.UNKNOWN
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
            self._input_name = self._session.get_inputs()[0].name
            self._output_kind = self._detect_output_kind()
            # Sanity-check the input dimension against FEATURE_ORDER so a
            # silently retrained model is reported instead of producing
            # garbage predictions on a shape mismatch.
            try:
                shape = self._session.get_inputs()[0].shape
                expected = len(FEATURE_ORDER)
                actual = shape[-1] if shape else None
                if isinstance(actual, int) and actual != expected:
                    logger.warning(
                        "Gesture MLP input dim %d != expected %d "
                        "(FEATURE_ORDER); refusing to load.",
                        actual,
                        expected,
                    )
                    self._session = None
                    self._loaded = False
                    return False
            except Exception:
                logger.debug("MLP: could not introspect input shape.", exc_info=True)
            self._loaded = True
            logger.info(
                "Gesture MLP loaded from %s (output_kind=%s)",
                path,
                self._output_kind.value,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load gesture MLP: %s", exc)
            self._loaded = False
            return False

    def _detect_output_kind(self) -> _OutputKind:
        """Introspect session output metadata to determine probability format."""
        try:
            outputs = self._session.get_outputs()
            if len(outputs) >= 2:
                out1 = outputs[1]
                # ZipMap output has type_str containing "map" or "Map".
                type_str = getattr(out1, "type", "") or ""
                if "map" in type_str.lower():
                    return _OutputKind.ZIPMAP
                # ndarray output: shape is [N, n_classes]
                return _OutputKind.NDARRAY_PROBS
        except Exception:
            pass
        return _OutputKind.UNKNOWN

    def _extract_probas(self, outputs: list[Any]) -> dict[str, float] | None:
        """Extract a label→probability mapping from raw ONNX outputs."""
        if (
            self._output_kind == _OutputKind.ZIPMAP
            and len(outputs) >= 2
            and isinstance(outputs[1], (list, tuple))
            and len(outputs[1]) > 0
        ):
            # outputs[1][0] is dict[str, float]
            probas = outputs[1][0]
            if isinstance(probas, dict):
                return {str(k): float(v) for k, v in probas.items()}
        elif self._output_kind == _OutputKind.NDARRAY_PROBS:
            # outputs[1] is a (N, n_classes) float32 array
            if len(outputs) >= 2:
                arr = outputs[1]
                if hasattr(arr, "shape") and arr.ndim >= 2:
                    row = arr[0]
                    if len(row) == len(MLP_GESTURE_LABELS):
                        return {
                            MLP_GESTURE_LABELS[i]: float(row[i])
                            for i in range(len(MLP_GESTURE_LABELS))
                        }
        # Fallback: try outputs[0] softmax
        if len(outputs) >= 1:
            arr = outputs[0]
            if hasattr(arr, "shape") and arr.ndim >= 2:
                row = arr[0]
                if len(row) == len(MLP_GESTURE_LABELS):
                    return {
                        MLP_GESTURE_LABELS[i]: float(row[i]) for i in range(len(MLP_GESTURE_LABELS))
                    }
        return None

    def close(self) -> None:
        """Release the ONNX session."""
        self._session = None
        self._loaded = False

    def classify(self, features: dict[str, Any] | None) -> tuple[str | None, float]:
        """Classify a feature dict.

        Returns:
            ``(label, confidence)`` or ``(None, 0.0)`` when classification
            is impossible (model not loaded, inference error, or bad output).
        """
        if not self._loaded or features is None or self._session is None:
            return (None, 0.0)

        vec = _features_to_vector(features)
        try:
            outputs = self._session.run(None, {self._input_name: vec})
            probas = self._extract_probas(outputs)
            if probas is None:
                logger.warning(
                    "MLP: could not parse output format (output_kind=%s, n_outputs=%d).",
                    self._output_kind.value,
                    len(outputs),
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


# ---------------------------------------------------------------------------
# Temporal TCN classifier
# ---------------------------------------------------------------------------


class TCNClassifier:
    """Temporal 1D-TCN gesture classifier using a sliding window of frames.

    Maintains an internal ring-buffer of ``window_size`` feature vectors.
    Each call to ``classify()`` appends one frame and runs ONNX inference
    when the buffer is full.  While warming up, ``(None, 0.0)`` is returned.

    If the ONNX model is absent at construction time, the classifier starts
    in disabled state and ``classify()`` always returns ``(None, 0.0)``.

    The classifier satisfies the :class:`~gazecontrol.gesture.classifier.GestureClassifier`
    Protocol so it can be used as the ``ml`` argument of
    :class:`~gazecontrol.gesture.fusion.GestureFusion`.
    """

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = window_size
        self._buffer: deque[list[float]] = deque(maxlen=window_size)
        self._session: Any = None
        self._input_name: str | None = None
        self._labels: list[str] = list(DEFAULT_LABELS)
        self._loaded = False
        self._try_load(str(Paths.gesture_tcn_model()))

    def _try_load(self, path: str) -> None:
        import os

        if not os.path.isfile(path):
            logger.debug("TCN model not found at %s; classifier disabled.", path)
            return
        self.load(path)

    def is_loaded(self) -> bool:
        """Return True when the ONNX session is ready."""
        return self._loaded

    def load(self, path: str, labels: list[str] | None = None) -> bool:
        """Load an ONNX model from *path*.  Returns True on success."""
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(path)
            self._input_name = self._session.get_inputs()[0].name
            self._labels = labels if labels is not None else self._load_labels_from_manifest(path)
            self._loaded = True
            logger.info("Gesture TCN loaded from %s (%d labels)", path, len(self._labels))
            return True
        except Exception as exc:
            logger.warning("Failed to load gesture TCN: %s", exc)
            self._loaded = False
            return False

    def _load_labels_from_manifest(self, model_path: str) -> list[str]:
        manifest_path = Path(model_path).parent / "manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text())
                name = Path(model_path).name
                if name in data:
                    return list(data[name].get("labels", DEFAULT_LABELS))
            except Exception:
                pass
        return list(DEFAULT_LABELS)

    def close(self) -> None:
        """Release the ONNX session and clear the buffer."""
        self._session = None
        self._loaded = False
        self._buffer.clear()

    def reset(self) -> None:
        """Clear the temporal buffer.  Call when the hand is lost between frames."""
        self._buffer.clear()

    def classify(self, features: FeatureSet | dict[str, Any] | None) -> tuple[str | None, float]:
        """Append one frame to the buffer and classify when the window is full.

        Args:
            features: Current frame's :class:`~gazecontrol.gesture.feature_extractor.FeatureSet`
                      (or legacy dict), or ``None`` when no hand is present.

        Returns:
            ``(label, confidence)`` once the buffer is full and the model is
            loaded; ``(None, 0.0)`` while warming up or when no hand.
        """
        if features is None:
            self._buffer.clear()
            return (None, 0.0)

        try:
            if hasattr(features, "to_vector"):
                vec: list[float] = features.to_vector()
            else:
                vec = list(_features_to_vector(features)[0])
        except (ValueError, TypeError, AttributeError) as exc:
            logger.debug("TCN feature vectorisation failed: %s", exc)
            return (None, 0.0)

        self._buffer.append(vec)

        if not self._loaded or len(self._buffer) < self._window_size or self._session is None:
            return (None, 0.0)

        return self._run_inference()

    def _run_inference(self) -> tuple[str | None, float]:
        try:
            arr = np.array(list(self._buffer), dtype=np.float32)
            x = arr[np.newaxis, :, :]
            outputs = self._session.run(None, {self._input_name: x})
            logits: np.ndarray[Any, Any] = outputs[0][0]
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            idx = int(np.argmax(probs))
            if idx < len(self._labels):
                return (self._labels[idx], float(probs[idx]))
            return (None, 0.0)
        except Exception as exc:
            logger.warning("TCN inference error: %s", exc)
            return (None, 0.0)
