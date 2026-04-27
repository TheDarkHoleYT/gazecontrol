"""Train and export the gesture MLP classifier to ONNX.

Usage::

    python tools/train_gesture_mlp.py \\
        --data gesture_data.csv \\
        --output models/gesture_mlp.onnx

The CSV must have columns matching the feature vector layout expected by
``gazecontrol.gesture.mlp_classifier._features_to_vector``:
    finger_states_0..4, finger_angles_0..4, palm_direction,
    hand_velocity_x, hand_velocity_y, thumb_index_distance, label
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MLP_GESTURE_LABELS = ["PINCH", "SWIPE_LEFT", "SWIPE_RIGHT", "MAXIMIZE"]


def train_and_export(X: np.ndarray, y: np.ndarray, output_path: str) -> None:
    """Train a scikit-learn MLP and export to ONNX.

    Args:
        X:           Feature matrix (N, F) float32.
        y:           Label array (N,) str — must be subset of MLP_GESTURE_LABELS.
        output_path: Destination ONNX file path.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.neural_network import MLPClassifier as SkMLP
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(MLP_GESTURE_LABELS)
    y_enc = le.transform(y)

    clf = SkMLP(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        activation="relu",
        solver="adam",
        random_state=42,
    )
    clf.fit(X, y_enc)
    logger.info(
        "MLP trained: %d samples, %d features, %d classes.", *X.shape, len(MLP_GESTURE_LABELS)
    )

    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
    options = {id(clf): {"zipmap": True}}
    onnx_model = convert_sklearn(clf, initial_types=initial_type, options=options)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())
    logger.info("Exported gesture MLP to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gesture MLP and export to ONNX.")
    parser.add_argument("--data", required=True, help="Path to CSV training data.")
    parser.add_argument("--output", default="models/gesture_mlp.onnx", help="Output ONNX path.")
    args = parser.parse_args()

    import csv

    rows: list[list[str]] = []
    with open(args.data, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    if not rows:
        logger.error("No data found in %s", args.data)
        return

    feature_cols = [k for k in rows[0] if k != "label"]
    X = np.array([[float(row[c]) for c in feature_cols] for row in rows], dtype=np.float32)
    y = np.array([row["label"] for row in rows])

    train_and_export(X, y, args.output)


if __name__ == "__main__":
    main()
