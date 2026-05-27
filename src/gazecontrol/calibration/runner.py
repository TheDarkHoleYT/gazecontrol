"""Qt-based L2CS gaze calibration runner.

Walks the user through a calibration grid of fixation targets (9, 13,
or a smaller incremental subset). For each target:

    1. Show a pulsing dot at the target screen position.
    2. Wait 1 s for the user to fixate.
    3. Capture *N* consecutive frames, run face crop + L2CS, store the
       resulting (yaw, pitch) angles paired with the target screen point.

After the grid completes, fit (or :meth:`partial_fit` for incremental
runs) a :class:`GazeMapper` and persist it to ``Paths.gaze_profile(profile)``.
For the full 13-point flow the 4 holdout targets are split out and the
mapper reports a real generalisation error alongside the LOO metric.

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

from gazecontrol.calibration.grid import (
    FULL_GRID,
    split_train_holdout,
    subset_targets,
)
from gazecontrol.errors import CalibrationError, ModelLoadError
from gazecontrol.i18n import t

logger = logging.getLogger(__name__)


_DWELL_S = 1.0
_CAPTURE_FRAMES = 20


@dataclass
class CalibrationResult:
    """Outcome of a calibration session."""

    success: bool
    loo_error_px: float = 0.0
    holdout_error_px: float | None = None
    points_captured: int = 0
    profile_path: str = ""
    fit_method: str = ""


def run_gaze_calibration(
    profile: str,
    vdesk: tuple[int, int, int, int],
    *,
    subset_size: int = 13,
    base_profile: str | None = None,
) -> int:
    """Run the calibration UI and persist the gaze profile.

    Args:
        profile:       Calibration profile name to write.
        vdesk:         (left, top, width, height) of the virtual desktop.
        subset_size:   How many grid points to capture this session
                       (3 / 5 / 9 / 13 — see :mod:`calibration.grid`).
                       13 is the default for a fresh calibration; smaller
                       values drive the incremental ``--calibrate-incremental``
                       top-up flow.
        base_profile:  When set, the runner loads the matching profile
                       first and calls :meth:`GazeMapper.partial_fit`
                       instead of :meth:`fit`. Combined with a small
                       ``subset_size`` this yields a quick "top-up"
                       recalibration that keeps the existing samples
                       and downweights nothing extra.

    Returns 0 on success, non-zero exit code on failure.
    """
    try:
        result = _run(
            profile=profile,
            vdesk=vdesk,
            subset_size=subset_size,
            base_profile=base_profile,
        )
    except CalibrationError as exc:
        print(f"Calibration error: {exc.user_message()}", file=sys.stderr)
        return 2
    except ModelLoadError as exc:
        print(f"Model load error: {exc.user_message()}", file=sys.stderr)
        return 3
    if not result.success:
        print("Calibration did not complete.", file=sys.stderr)
        return 1
    msg = (
        f"Calibration saved to {result.profile_path} "
        f"(LOO error ≈ {result.loo_error_px:.1f} px"
    )
    if result.holdout_error_px is not None:
        msg += f", holdout error ≈ {result.holdout_error_px:.1f} px"
    msg += f", method={result.fit_method})"
    print(msg)
    return 0


def _run(
    profile: str,
    vdesk: tuple[int, int, int, int],
    *,
    subset_size: int = 13,
    base_profile: str | None = None,
) -> CalibrationResult:
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

    # Subset selection (G8b). For non-13 sizes the holdout split is
    # skipped (the runner reports LOO only); for 13 the four holdout
    # targets are tagged via their FULL_GRID index and pulled aside
    # after capture for compute_holdout_error().
    grid_targets = subset_targets(subset_size)
    # If the subset is the full grid we know the FULL_GRID indices and
    # can compute holdout error; otherwise tag all frames as "train".
    target_indices_in_full_grid: list[int] = []
    if subset_size == 13:
        target_indices_in_full_grid = list(range(13))
    else:
        # Map each captured target back to FULL_GRID for telemetry.
        target_indices_in_full_grid = [
            FULL_GRID.index(t) if t in FULL_GRID else -1 for t in grid_targets
        ]

    captured_angles: list[tuple[float, float]] = []
    captured_targets: list[tuple[int, int]] = []
    # Per-frame "which FULL_GRID index" so we can split train / holdout
    # after the grid completes (only meaningful when subset_size==13).
    captured_target_indices: list[int] = []

    app = QApplication.instance() or QApplication(sys.argv)

    class _CalWindow(QWidget):
        def __init__(self) -> None:
            super().__init__()
            # Initialise all instance attributes BEFORE showing the window —
            # setGeometry/showFullScreen can fire showEvent which reads them.
            self._point_index = 0
            self._dwell_started: float | None = None
            self._captures_for_current = 0
            self._target_norm: tuple[float, float] = grid_targets[0]
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
            if self._point_index >= len(grid_targets):
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
            captured_target_indices.append(
                target_indices_in_full_grid[self._point_index]
            )
            self._captures_for_current += 1
            if self._captures_for_current >= _CAPTURE_FRAMES:
                self._captures_for_current = 0
                self._dwell_started = None
                self._point_index += 1
                if self._point_index < len(grid_targets):
                    self._target_norm = grid_targets[self._point_index]
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
            total = len(grid_targets)
            p.setPen(QPen(QColor(220, 220, 220, 220)))
            p.drawText(
                QPointF(40, 40), t("calibration.progress", done=done, total=total)
            )
            p.drawText(QPointF(40, 60), t("calibration.instructions"))
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

    # --- G8b: train / holdout split (only when subset_size == 13) ------
    train_idx, holdout_idx = split_train_holdout(captured_target_indices)
    if subset_size != 13:
        # No holdout for incremental / 9-pt runs — all samples train.
        train_idx = list(range(len(angles_arr)))
        holdout_idx = []
    train_angles = angles_arr[train_idx] if train_idx else angles_arr
    train_targets = targets_arr[train_idx] if train_idx else targets_arr
    holdout_angles = angles_arr[holdout_idx] if holdout_idx else None
    holdout_targets = targets_arr[holdout_idx] if holdout_idx else None

    # --- G8a + G8b: incremental refit vs full fit ----------------------
    mapper = GazeMapper(screen_w=cal_w, screen_h=cal_h)
    fit_method = "13pt_holdout" if subset_size == 13 else f"{subset_size}pt"
    holdout_error_px: float | None = None
    if base_profile is not None:
        # Incremental "top-up" — load existing profile, partial_fit on
        # the new train samples. Skip holdout for partial fits.
        base_path = Paths.gaze_profile(base_profile)
        if not base_path.exists():
            raise CalibrationError(
                f"--calibrate-incremental requires --profile {base_profile!r} "
                f"to already exist at {base_path}"
            )
        if not mapper.load(base_path):
            raise CalibrationError(
                f"Failed to load base profile from {base_path}"
            )
        try:
            loo = mapper.partial_fit(
                train_angles,
                train_targets,
                fit_method=f"incremental_{subset_size}pt",
            )
        except (RuntimeError, ValueError) as exc:
            raise CalibrationError(f"partial_fit failed: {exc}") from exc
        fit_method = f"incremental_{subset_size}pt"
    else:
        try:
            loo = mapper.fit(train_angles, train_targets, fit_method=fit_method)
        except Exception as exc:
            raise CalibrationError(f"GazeMapper.fit failed: {exc}") from exc
        if holdout_angles is not None and holdout_targets is not None:
            holdout_error_px = mapper.compute_holdout_error(
                holdout_angles, holdout_targets
            )
            # Persist the holdout metric into the schema-v2 metadata via
            # a tiny refit-of-the-same-data — the fit() API takes
            # holdout_error_px as a kwarg and stores it. Cheaper than
            # opening a private setter for one field.
            mapper.fit(
                train_angles,
                train_targets,
                fit_method=fit_method,
                holdout_error_px=holdout_error_px,
            )

    # G19 + G21: write to the v2 layout (<profiles>/<user>/<monitor>/v{N}.npz)
    # and update the latest.txt pointer. The monitor id is derived from
    # the Qt screen the calibration window actually rendered on, so a
    # session calibrated against a 4K external lands in its own bucket
    # separate from the laptop panel. The legacy flat path is mirrored
    # for backward-compat readers.
    from gazecontrol.gaze.monitor_id import (
        DEFAULT_MONITOR_ID,
        monitor_id_from_screen_info,
    )

    user_id = s.gaze.user_id
    try:
        from PyQt6.QtGui import QGuiApplication

        win_screen = window.screen() if hasattr(window, "screen") else None
        if win_screen is None:
            win_screen = QGuiApplication.primaryScreen()
        if win_screen is not None:
            wg = win_screen.geometry()
            monitor_id = monitor_id_from_screen_info(
                win_screen.name(),
                (int(wg.x()), int(wg.y()), int(wg.width()), int(wg.height())),
            )
        else:
            monitor_id = DEFAULT_MONITOR_ID
    except (ImportError, RuntimeError):
        monitor_id = DEFAULT_MONITOR_ID
    next_version = Paths.next_v2_profile_version(user_id, monitor_id)
    v2_path = Paths.gaze_profile_v2(user_id, monitor_id, next_version)
    mapper.set_profile_identity(user_id=user_id, monitor_id=monitor_id)
    mapper.save(v2_path.with_suffix(""))
    Paths.write_latest_pointer(user_id, monitor_id, next_version)
    # Mirror to the legacy flat path for backward-compat readers.
    profile_path = Paths.gaze_profile(profile)
    mapper.save(profile_path.with_suffix(""))
    logger.info(
        "Calibration: %d samples (train), %d holdout, LOO=%.1f px, "
        "holdout=%s, method=%s → %s",
        len(train_idx) or len(angles_arr),
        len(holdout_idx),
        loo,
        f"{holdout_error_px:.1f} px" if holdout_error_px is not None else "n/a",
        fit_method,
        profile_path,
    )
    return CalibrationResult(
        success=True,
        loo_error_px=loo,
        holdout_error_px=holdout_error_px,
        fit_method=fit_method,
        points_captured=len(captured_angles),
        profile_path=str(profile_path),
    )
