"""
Strumento diagnostico: verifica camera + face detection prima della calibrazione.

Uso:
    python tools/diagnose_calibration.py

Output:
  - Salva frame campione in tools/diag_frame_*.png per ispezione visiva
  - Riporta brightness, shape, esito face detection per ogni frame
  - Confronta camera senza config vs. con config (risoluzione + auto-exposure)
"""

import os
import sys

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _open_raw(index=0):
    """Apre camera senza configurazione (come faceva il vecchio codice)."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    cap.release()
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    cap.release()
    raise RuntimeError(f"Impossibile aprire camera {index}")


def _open_configured(index=0):
    """Apre camera con configurazione completa (come FrameGrabber)."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Impossibile aprire camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    return cap


def _capture_frames(cap, n=30, label=""):
    """Cattura n frame e ritorna lista di (ret, frame, brightness)."""
    results = []
    for i in range(n):
        ret, frame = cap.read()
        brightness = float(frame.mean()) if ret else 0.0
        shape = frame.shape if ret else None
        results.append((i, ret, frame if ret else None, brightness, shape))
    return results


def _test_mediapipe(frames, label=""):
    """Testa extract_features su una lista di (i, ret, frame, brightness, shape)."""
    from eyetrax import GazeEstimator

    print(f"\n  [MediaPipe {label}] Inizializzazione GazeEstimator...")
    try:
        estimator = GazeEstimator(
            model_name="tiny_mlp",
            model_kwargs={"hidden_layer_sizes": (256, 128, 64), "max_iter": 500},
        )
    except Exception as e:
        print(f"  ERRORE init GazeEstimator: {e}")
        return

    detected = 0
    for i, ret, frame, brightness, _shape in frames:
        if not ret or frame is None:
            continue
        try:
            features, blink = estimator.extract_features(frame)
            face_ok = features is not None
            if face_ok:
                detected += 1
            if i in (0, 5, 15, 29):
                status = "FACCIA OK" if face_ok else "NO FACCIA"
                print(f"    frame {i:2d}: {status}, blink={blink}, brightness={brightness:.1f}")
        except Exception as e:
            print(f"    frame {i:2d}: ECCEZIONE {e}")

    print(f"  Facce rilevate: {detected}/{len(frames)} frame")


def run_test(label, open_fn, save_prefix, n=30):
    print(f"\n{'=' * 60}")
    print(f"TEST: {label}")
    print(f"{'=' * 60}")

    try:
        cap = open_fn()
    except RuntimeError as e:
        print(f"  ERRORE apertura camera: {e}")
        return []

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Risoluzione: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    frames = _capture_frames(cap, n=n, label=label)
    cap.release()

    out_dir = os.path.join(os.path.dirname(__file__))
    ok_count = sum(1 for _, ret, _, _, _ in frames if ret)
    print(f"  Frame letti OK: {ok_count}/{n}")

    # Salva frame campione
    for i, ret, frame, brightness, shape in frames:
        if ret and i in (0, 5, 15, n - 1):
            path = os.path.join(out_dir, f"{save_prefix}_{i}.png")
            cv2.imwrite(path, frame)
            print(f"  Salvato: {path}  (brightness={brightness:.1f}, shape={shape})")

    # Brightness stats
    brightnesses = [b for _, ret, _, b, _ in frames if ret]
    if brightnesses:
        print(
            f"  Brightness: min={min(brightnesses):.1f}  mean={sum(brightnesses) / len(brightnesses):.1f}  max={max(brightnesses):.1f}"
        )

    # Test MediaPipe
    valid_frames = [(i, ret, frame, b, s) for i, ret, frame, b, s in frames if ret]
    _test_mediapipe(valid_frames, label=label)

    return frames


def main():
    print("GazeControl - Diagnostica Calibrazione")
    print(f"Python: {sys.version}")

    try:
        import mediapipe as mp

        print(f"MediaPipe: {mp.__version__}")
    except ImportError:
        print("MediaPipe: NON INSTALLATO")

    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")

    # Test 1: camera senza configurazione (comportamento vecchio)
    run_test(
        label="CAMERA RAW (senza configurazione)",
        open_fn=_open_raw,
        save_prefix="diag_raw",
        n=30,
    )

    # Test 2: camera con configurazione completa
    run_test(
        label="CAMERA CONFIGURATA (1280x720, no auto-exposure, warmup)",
        open_fn=lambda: _open_configured(0),
        save_prefix="diag_configured",
        n=30,
    )

    print(f"\n{'=' * 60}")
    print("Controlla i file diag_*.png in tools/ per ispezione visiva.")
    print("Se 'CAMERA CONFIGURATA' mostra piu' facce rilevate, il fix e' corretto.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
