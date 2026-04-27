"""Qt-based L2CS gaze calibration runner.

Walks the user through a 3×3 grid of fixation targets. For each target:

    1. Show a pulsing dot at the target screen position.
    2. Wait 1 s for the user to fixate.
    3. Capture *N* consecutive frames, run face crop + L2CS, store the
       resulting (yaw, pitch) angles paired with the target screen point.

After the grid completes, fit a :class:`GazeMapper` and persist it to
``Paths.gaze_profile(profile)``. The leave-one-out error is logged and
returned to the caller.

This runner uses the existing :class:`FrameGrabber` (no fresh
``cv2.VideoCapture``) so it cannot conflict with a running pipeline.
The pipeline must be stopped before calling ``run_gaze_calibration``.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass
from typing import Any

from gazecontrol.errors import CalibrationError, ModelLoadError

logger = logging.getLogger(__name__)


_GRID = [
    (0.10, 0.10),
    (0.50, 0.10),
    (0.90, 0.10),
    (0.10, 0.50),
    (0.50, 0.50),
    (0.90, 0.50),
    (0.10, 0.90),
    (0.50, 0.90),
    (0.90, 0.90),
]
_DWELL_S = 1.0
_CAPTURE_FRAMES = 20


@dataclass
class CalibrationResult:
    """Outcome of a calibration session."""

    success: bool
    loo_error_px: float = 0.0
    points_captured: int = 0
    profile_path: str = ""


def run_gaze_calibration(
    profile: str,
    vdesk: tuple[int, int, int, int],
) -> int:
    """Run the calibration UI and persist the gaze profile.

    Returns 0 on success, non-zero exit code on failure.
    """
    try:
        result = _run(profile=profile, vdesk=vdesk)
    except CalibrationError as exc:
        print(f"Calibration error: {exc.user_message()}", file=sys.stderr)
        return 2
    except ModelLoadError as exc:
        print(f"Model load error: {exc.user_message()}", file=sys.stderr)
        return 3
    if not result.success:
        print("Calibration did not complete.", file=sys.stderr)
        return 1
    print(f"Calibration saved to {result.profile_path} (LOO error ≈ {result.loo_error_px:.1f} px)")
    return 0


def _run(profile: str, vdesk: tuple[int, int, int, int]) -> CalibrationResult:
    import numpy as np

    try:
        from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
        from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
        from PyQt6.QtWidgets import QApplication, QWidget
    except ImportError as exc:
        raise CalibrationError("PyQt6 is required for calibration UI.") from exc

    from gazecontrol.capture.frame_grabber import FrameGrabber
    from gazecontrol.gaze.face_crop import FaceCropper
    from gazecontrol.gaze.gaze_mapper import GazeMapper
    from gazecontrol.gaze.l2cs_model import L2CSModel
    from gazecontrol.paths import Paths
    from gazecontrol.settings import get_settings

    s = get_settings()
    left, top, width, height = vdesk

    model_path = Paths.l2cs_model()
    if not model_path.exists():
        raise ModelLoadError(f"L2CS model missing at {model_path}")
    try:
        model = L2CSModel(str(model_path))
    except Exception as exc:
        raise ModelLoadError(str(exc)) from exc
    cropper = FaceCropper()
    if not model.is_loaded:
        raise ModelLoadError("L2CS model failed to initialise.")

    grabber = FrameGrabber(
        camera_index=s.camera.index,
        width=s.camera.width,
        height=s.camera.height,
        fps=s.camera.fps,
    )
    if not grabber.start():
        raise CalibrationError("Camera failed to start; close other apps using it.")

    captured_angles: list[tuple[float, float]] = []
    captured_targets: list[tuple[int, int]] = []

    app = QApplication.instance() or QApplication(sys.argv)

    class _CalWindow(QWidget):
        def __init__(self) -> None:
            super().__init__()
            # Initialise all instance attributes BEFORE showing the window —
            # setGeometry/showFullScreen can fire showEvent which reads them.
            self._point_index = 0
            self._dwell_started: float | None = None
            self._captures_for_current = 0
            self._target_norm: tuple[float, float] = _GRID[0]
            self._target_screen: tuple[int, int] = (0, 0)

            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
            )
            self.setStyleSheet("background-color: rgb(15, 18, 26);")
            # Pin to the *primary* screen geometry (not the virtual desktop).
            # showFullScreen() only stretches across one monitor, so building
            # target points on virtual-desktop coords would land them off-screen
            # on multi-monitor setups.
            try:
                from PyQt6.QtGui import QGuiApplication

                primary = QGuiApplication.primaryScreen()
                if primary is not None:
                    geo = primary.geometry()
                    self.setGeometry(geo)
                    wh = self.windowHandle()
                    if wh is not None:
                        wh.setScreen(primary)
            except Exception:
                self.setGeometry(left, top, width, height)
            self.showFullScreen()
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(33)

        def _screen_pos(self, normalized: tuple[float, float]) -> tuple[int, int]:
            """Map a normalised target to local widget coords (pixel-safe)."""
            nx, ny = normalized
            margin = 60  # keep targets clear of the bezel
            w = max(self.width(), 1)
            h = max(self.height(), 1)
            inner_w = max(w - 2 * margin, 1)
            inner_h = max(h - 2 * margin, 1)
            x = margin + int(nx * inner_w)
            y = margin + int(ny * inner_h)
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            return (x, y)

        def showEvent(self, event: Any) -> None:  # noqa: N802
            super().showEvent(event)
            # Recompute the first target now that we have a real widget size.
            self._target_screen = self._screen_pos(self._target_norm)

        def _tick(self) -> None:
            if self._point_index >= len(_GRID):
                self.close()
                return
            now = time.monotonic()
            if self._dwell_started is None:
                self._dwell_started = now
                return
            if (now - self._dwell_started) < _DWELL_S:
                self.update()
                return
            ok, frame_bgr = grabber.read_bgr()
            if not ok or frame_bgr is None:
                return
            crop = cropper.crop_from_frame(frame_bgr)
            if crop is None:
                return
            angles = model.predict(crop)
            if angles is None:
                return
            captured_angles.append(angles)
            captured_targets.append(self._target_screen)
            self._captures_for_current += 1
            if self._captures_for_current >= _CAPTURE_FRAMES:
                self._captures_for_current = 0
                self._dwell_started = None
                self._point_index += 1
                if self._point_index < len(_GRID):
                    self._target_norm = _GRID[self._point_index]
                    self._target_screen = self._screen_pos(self._target_norm)
            self.update()

        def paintEvent(self, _event: object) -> None:  # noqa: N802
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            now = time.monotonic()
            tx, ty = self._target_screen
            pulse = 6 + 4 * math.sin(now * 6)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(0, 220, 100, 220)))
            p.drawEllipse(QRectF(tx - pulse, ty - pulse, pulse * 2, pulse * 2))
            p.setPen(QPen(QColor(255, 255, 255, 200), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(tx, ty), 28, 28)
            done = self._point_index
            total = len(_GRID)
            p.setPen(QPen(QColor(220, 220, 220, 220)))
            p.drawText(QPointF(40, 40), f"Calibrazione gaze: {done}/{total}")
            p.drawText(QPointF(40, 60), "Fissa il punto verde, tieni la testa ferma.")
            p.end()

    window = _CalWindow()
    app.exec()
    grabber.stop()

    if len(captured_angles) < 5 * _CAPTURE_FRAMES:
        return CalibrationResult(
            success=False,
            points_captured=len(captured_angles),
        )

    # Targets were captured in the calibration window's local coords
    # (primary screen). Train and persist the mapper using *those* coords
    # so predict() returns values in the same space at runtime.
    cal_w = max(window.width(), 1)
    cal_h = max(window.height(), 1)
    angles_arr = np.asarray(captured_angles, dtype=float)
    targets_arr = np.asarray(captured_targets, dtype=float)
    mapper = GazeMapper(screen_w=cal_w, screen_h=cal_h)
    try:
        loo = mapper.fit(angles_arr, targets_arr)
    except Exception as exc:
        raise CalibrationError(f"GazeMapper.fit failed: {exc}") from exc

    profile_path = Paths.gaze_profile(profile)
    mapper.save(profile_path.with_suffix(""))  # GazeMapper.save adds .npz/.meta.json
    logger.info(
        "Calibration: %d samples, LOO error = %.1f px → %s",
        len(captured_angles),
        loo,
        profile_path,
    )
    return CalibrationResult(
        success=True,
        loo_error_px=loo,
        points_captured=len(captured_angles),
        profile_path=str(profile_path),
    )
