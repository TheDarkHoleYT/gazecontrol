"""record_gesture_dataset.py — interactive gesture data collection tool.

Records labelled hand-gesture samples for training the temporal TCN classifier.
Each sample is a window of 30 consecutive frames, storing the 16-element feature
vector per frame (shape: [30, 16]).

Usage::

    python tools/record_gesture_dataset.py --label PINCH --samples 100 --out data/gestures/
    python tools/record_gesture_dataset.py --label GRAB --samples 50 --out data/gestures/

The output directory receives one ``.npz`` file per recorded sample::

    data/gestures/PINCH_001.npz
    data/gestures/PINCH_002.npz
    ...

Each file contains:
- ``features``: float32 array of shape ``[T, F]`` (T frames, F features).
- ``label``: str scalar (e.g. ``"PINCH"``).
- ``timestamp``: float64 Unix timestamp.
- ``hand_length_norm``: float32 mean normalised hand length across the window.

A manifest ``data/gestures/manifest.json`` is updated with per-label counts.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="record_gesture_dataset",
        description="Record labelled gesture samples for TCN training.",
    )
    p.add_argument(
        "--label",
        required=True,
        help=(
            "Gesture label to record (e.g. PINCH, GRAB, SCROLL_UP). "
            "Must match a GestureLabel enum value."
        ),
    )
    p.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to record (default: 100).",
    )
    p.add_argument(
        "--out",
        default="data/gestures",
        help="Output directory for .npz files (default: data/gestures).",
    )
    p.add_argument(
        "--window",
        type=int,
        default=30,
        help="Number of frames per sample window (default: 30).",
    )
    p.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index override (uses settings default when omitted).",
    )
    p.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Countdown seconds before each recording burst (default: 3).",
    )
    return p


def _validate_label(label: str) -> str:
    from gazecontrol.gesture.labels import GestureLabel

    try:
        return GestureLabel(label).value
    except ValueError:
        valid = ", ".join(gl.value for gl in GestureLabel)
        print(f"[ERROR] Unknown label '{label}'. Valid labels: {valid}", file=sys.stderr)
        sys.exit(1)


def _update_manifest(out_dir: Path, label: str, count: int) -> None:
    manifest_path = out_dir / "manifest.json"
    manifest: dict[str, int] = {}
    if manifest_path.exists():
        with contextlib.suppress(Exception):
            manifest = json.loads(manifest_path.read_text())
    manifest[label] = manifest.get(label, 0) + count
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    args = _build_argparser().parse_args()
    label = _validate_label(args.label)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[ERROR] opencv-python and numpy are required.", file=sys.stderr)
        sys.exit(1)

    from gazecontrol.gesture.feature_extractor import GestureFeatureExtractor
    from gazecontrol.gesture.hand_detector import HandDetector
    from gazecontrol.settings import get_settings

    s = get_settings()
    cam_idx = args.camera if args.camera is not None else s.camera.index

    print(f"\n[record_gesture_dataset] Label: {label}")
    print(f"  Samples: {args.samples}  |  Window: {args.window} frames  |  Camera: {cam_idx}")
    print(f"  Output: {out_dir.resolve()}\n")

    # Open camera.
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam_idx}.", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.camera.height)
    cap.set(cv2.CAP_PROP_FPS, s.camera.fps)

    try:
        detector = HandDetector()
    except Exception as exc:
        print(f"[ERROR] HandDetector init failed: {exc}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    extractor = GestureFeatureExtractor()

    # Figure out existing file indices for this label.
    existing = sorted(out_dir.glob(f"{label}_*.npz"))
    next_idx = len(existing) + 1

    recorded = 0

    try:
        while recorded < args.samples:
            # Countdown before recording.
            print(f"\n[{recorded + 1}/{args.samples}] Show '{label}' gesture in:")
            for c in range(args.countdown, 0, -1):
                print(f"  {c}...", end=" ", flush=True)
                time.sleep(1.0)
            print("GO!")

            # Collect one window of frames.
            frame_features: list[list[float]] = []
            hand_lengths: list[float] = []
            ts_ms = 0

            while len(frame_features) < args.window:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = int(time.monotonic() * 1000)

                hand_result = detector.process(frame_rgb, ts_ms)
                if hand_result is None or not hand_result.multi_hand_landmarks:
                    print("  (waiting for hand…)", end="\r", flush=True)
                    continue

                feat = extractor.extract(hand_result)
                if feat is None:
                    continue

                frame_features.append(feat.to_vector())

                # Approximate hand length for normalisation metadata.
                lm = hand_result.multi_hand_landmarks[0].landmark
                import math

                hl = math.sqrt((lm[0].x - lm[9].x) ** 2 + (lm[0].y - lm[9].y) ** 2)
                hand_lengths.append(hl)

            # Save sample.
            out_path = out_dir / f"{label}_{next_idx:04d}.npz"
            feat_arr = np.array(frame_features, dtype=np.float32)  # [T, F]
            mean_hl = float(np.mean(hand_lengths)) if hand_lengths else 0.0
            np.savez_compressed(
                str(out_path),
                features=feat_arr,
                label=np.array(label),
                timestamp=np.array(time.time()),
                hand_length_norm=np.array(mean_hl, dtype=np.float32),
            )
            print(f"  Saved: {out_path.name}  shape={feat_arr.shape}")
            next_idx += 1
            recorded += 1

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        cap.release()
        detector.close()

    if recorded > 0:
        _update_manifest(out_dir, label, recorded)
        manifest_path = out_dir / "manifest.json"
        print(f"\n[done] Recorded {recorded} samples → {out_dir.resolve()}")
        print(f"  Manifest: {manifest_path}")
    else:
        print("\n[done] No samples recorded.")


if __name__ == "__main__":
    main()
