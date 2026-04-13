import logging
import os
import numpy as np
import gazecontrol.config as config

logger = logging.getLogger(__name__)

MLP_GESTURE_LABELS = ['PINCH', 'SWIPE_LEFT', 'SWIPE_RIGHT', 'MAXIMIZE']


def _features_to_vector(features: dict) -> np.ndarray:
    vec = []
    vec.extend([float(x) for x in features['finger_states']])
    vec.extend([float(x) for x in features['finger_angles']])
    vec.append(float(features['palm_direction']))
    vec.append(float(features['hand_velocity_x']))
    vec.append(float(features['hand_velocity_y']))
    vec.append(float(features['thumb_index_distance']))
    return np.array(vec, dtype=np.float32).reshape(1, -1)


class MLPClassifier:
    def __init__(self):
        self._session = None
        self._input_name = None
        self._loaded = False
        self._try_load(config.MLP_MODEL_PATH)

    def _try_load(self, path: str):
        if not os.path.isfile(path):
            return
        self.load(path)

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, path: str) -> bool:
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(path)
            self._input_name = self._session.get_inputs()[0].name
            self._loaded = True
            logger.info("Loaded gesture MLP from %s", path)
            return True
        except Exception as e:
            logger.warning("Failed to load ONNX model: %s", e)
            self._loaded = False
            return False

    def classify(self, features: dict) -> tuple:
        if not self._loaded or features is None:
            return (None, 0.0)

        vec = _features_to_vector(features)
        try:
            outputs = self._session.run(None, {self._input_name: vec})
            probas = outputs[1][0]
            if not isinstance(probas, dict):
                logger.warning("MLP returned non-dict probabilities (type=%s); skipping", type(probas).__name__)
                return (None, 0.0)
            missing = [k for k in MLP_GESTURE_LABELS if k not in probas]
            if missing:
                logger.warning("MLP probabilities missing keys: %s", missing)
                return (None, 0.0)
            idx = int(np.argmax([probas[label] for label in MLP_GESTURE_LABELS]))
            label = MLP_GESTURE_LABELS[idx]
            confidence = float(probas[label])
            return (label, confidence)
        except Exception as e:
            logger.warning("MLP inference error: %s", e)
            return (None, 0.0)


def train_and_export(X, y, output_path: str):
    from sklearn.neural_network import MLPClassifier as SkMLP
    from sklearn.preprocessing import LabelEncoder
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    le = LabelEncoder()
    le.fit(MLP_GESTURE_LABELS)
    y_enc = le.transform(y)

    clf = SkMLP(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        activation='relu',
        solver='adam',
        random_state=42,
    )
    clf.fit(X, y_enc)

    initial_type = [('input', FloatTensorType([None, X.shape[1]]))]
    options = {id(clf): {'zipmap': True}}
    onnx_model = convert_sklearn(clf, initial_types=initial_type, options=options)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    logger.info("Exported MLP to %s", output_path)
