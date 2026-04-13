"""Calibration helpers — extracted from main.py.

Contains:
- open_camera_dshow()  : DirectShow camera opener for Windows.
- wait_for_face()      : Improved face-wait UI for eyetrax calibration.
- run_calibration()    : Top-level calibration entry point.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from gazecontrol.paths import Paths
from gazecontrol.settings import get_settings

logger = logging.getLogger(__name__)


def open_camera_dshow(index: int = 0) -> cv2.VideoCapture:
    """Open camera with DirectShow backend on Windows (best for eyetrax).

    Tries candidates in priority order and returns the first that opens.
    Sets resolution, FPS, disables auto-exposure, and warms up the capture.
    """
    s = get_settings().camera
    candidates = [
        (index, cv2.CAP_DSHOW),
        (index, cv2.CAP_ANY),
    ]
    fallback = 1 if index == 0 else 0
    candidates += [(fallback, cv2.CAP_DSHOW), (fallback, cv2.CAP_ANY)]

    cap: cv2.VideoCapture | None = None
    for cam_idx, backend in candidates:
        _cap = cv2.VideoCapture(cam_idx, backend)
        if _cap.isOpened():
            cap = _cap
            break
        _cap.release()

    if cap is None:
        raise RuntimeError(f"Cannot open camera {index} (tried all backends)")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
    cap.set(cv2.CAP_PROP_FPS, s.fps)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    for _ in range(s.warmup_frames):
        cap.read()

    return cap


def wait_for_face(
    cap: cv2.VideoCapture,
    gaze_estimator: object,
    sw: int,
    sh: int,
    dur: int = 2,
) -> bool:
    """Wait until a face is detected stably for *dur* seconds.

    Draws a dark-grey background (reduces webcam over-exposure), a small
    camera preview in the corner, and a countdown arc.

    Returns True when face detected stably; False on timeout (60 s) or ESC.
    """
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fd_start: float | None = None
    countdown = False
    attempt_start = time.monotonic()
    max_wait_s = 60
    consecutive_failures = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= 30:
                logger.error("Camera stopped returning frames during face-wait.")
                return False
            if time.monotonic() - attempt_start > max_wait_s:
                logger.error("Timeout waiting for valid frame from camera.")
                return False
            continue
        consecutive_failures = 0

        f, blink = gaze_estimator.extract_features(frame)  # type: ignore[union-attr]
        face = f is not None and not blink

        canvas = np.full((sh, sw, 3), 30, dtype=np.uint8)

        # Camera preview (320×180) bottom-right corner.
        try:
            preview = cv2.resize(frame, (320, 180))
            canvas[sh - 190 : sh - 10, sw - 330 : sw - 10] = preview
            color = (0, 200, 0) if face else (0, 0, 200)
            cv2.rectangle(canvas, (sw - 331, sh - 191), (sw - 9, sh - 9), color, 2)
        except Exception:  # noqa: BLE001
            pass

        now = time.monotonic()
        elapsed_total = now - attempt_start

        if face:
            if not countdown:
                fd_start = now
                countdown = True
            elapsed = now - (fd_start or now)
            if elapsed >= dur:
                return True
            t = elapsed / dur
            e = t * t * (3 - 2 * t)
            ang = 360 * (1 - e)
            cv2.ellipse(canvas, (sw // 2, sh // 2), (50, 50), 0, -90, -90 + ang, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                "Faccia rilevata - resta fermo",
                (sw // 2 - 280, sh // 2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
        else:
            countdown = False
            fd_start = None

            if elapsed_total > max_wait_s:
                logger.error(
                    "Timeout face detection after %.0f s. "
                    "Check lighting and webcam position.",
                    elapsed_total,
                )
                return False

            txt = "Faccia non rilevata"
            fs, thick = 2, 3
            size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            tx = (sw - size[0]) // 2
            ty = (sh + size[1]) // 2 - 60
            cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), thick)
            hint = f"Assicurati di essere davanti alla webcam ({int(max_wait_s - elapsed_total)}s)"
            cv2.putText(
                canvas,
                hint,
                (sw // 2 - 350, ty + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 180, 180),
                1,
            )

        cv2.imshow("Calibration", canvas)
        if cv2.waitKey(1) == 27:
            return False


def run_calibration(profile_name: str = "default", adaptive: bool = False) -> None:
    """Run eyetrax calibration and save profile.

    Args:
        profile_name: Name of the profile to save (stored in platformdirs config dir).
        adaptive:     If True, use 9+60 adaptive calibration (~2.5 min).
                      If False, use 5×5 dense grid (~50 s).
    """
    from eyetrax import GazeEstimator
    from eyetrax.calibration import run_dense_grid_calibration

    from gazecontrol.gaze.compat.eyetrax import apply_patches, PatchError

    profile_dir = Paths.profiles()
    profile_path = profile_dir / f"{profile_name}.pkl"

    try:
        apply_patches(open_camera_dshow, wait_for_face)
    except PatchError as exc:
        logger.warning("eyetrax patch failed: %s — calibration may hang on Windows.", exc)

    estimator = GazeEstimator(
        model_name="tiny_mlp",
        model_kwargs={
            "hidden_layer_sizes": (256, 128, 64),
            "max_iter": 1000,
            "alpha": 1e-4,
            "early_stopping": True,
        },
    )

    if adaptive:
        from eyetrax.calibration.adaptive import run_adaptive_calibration

        logger.info("Starting ADAPTIVE calibration (9 + 60 points)...")
        run_adaptive_calibration(estimator, num_random_points=60, retrain_every=10)
    else:
        logger.info("Starting dense 5×5 grid calibration (25 points)...")
        run_dense_grid_calibration(estimator, rows=5, cols=5, order="serpentine")

    estimator.save_model(str(profile_path))
    logger.info("Profile '%s' saved to %s", profile_name, profile_path)
