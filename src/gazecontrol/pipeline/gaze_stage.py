"""GazeStage — gaze estimation, filtering, L2CS ensemble, fixation detection.

Bug fixes applied here vs the old main.py:
- L2CS bridge: crop_from_landmarks() called with real MediaPipe landmarks.
- GazeMapper: predict() returning None is handled (ensemble weight renormalised).
- L2CSModel.enabled flag checked before calling predict().
- time.monotonic() used exclusively.
- blink_hold_max_s is a named setting, not a magic literal.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from gazecontrol.gaze.drift_corrector import DriftCorrector
from gazecontrol.gaze.face_crop import FaceCropper
from gazecontrol.gaze.fixation_detector import FixationDetector
from gazecontrol.gaze.gaze_mapper import GazeMapper
from gazecontrol.gaze.one_euro_filter import OneEuroFilter
from gazecontrol.paths import Paths
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.settings import get_settings

logger = logging.getLogger(__name__)


class GazeStage:
    """Gaze estimation pipeline stage.

    Responsibilities:
    - Run eyetrax TinyMLP to get raw screen gaze coords.
    - Optionally run L2CS-Net CNN and blend into ensemble.
    - Apply OneEuroFilter for adaptive smoothing.
    - Apply DriftCorrector for long-term stability.
    - Classify each sample as fixation / saccade / pursuit via I-VT.
    """

    def __init__(self, screen_w: int, screen_h: int) -> None:
        s = get_settings()
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._gaze_cfg = s.gaze
        self._fix_cfg = s.fixation

        # OneEuroFilter — one per axis.
        self._filter_x = OneEuroFilter(
            freq=s.camera.fps,
            min_cutoff=s.gaze.one_euro_min_cutoff,
            beta=s.gaze.one_euro_beta,
        )
        self._filter_y = OneEuroFilter(
            freq=s.camera.fps,
            min_cutoff=s.gaze.one_euro_min_cutoff,
            beta=s.gaze.one_euro_beta,
        )
        self._fixation_detector = FixationDetector(
            screen_px_per_degree=s.fixation.px_per_deg,
            fixation_vel_thr=s.fixation.velocity_threshold_deg_s,
            saccade_vel_thr=s.fixation.saccade_threshold_deg_s,
        )
        self._drift_corrector = DriftCorrector(
            screen_w=screen_w, screen_h=screen_h
        )
        self._gaze_mapper = GazeMapper(screen_w=screen_w, screen_h=screen_h)
        self._face_cropper = FaceCropper()

        # L2CS-Net — loaded lazily, disabled gracefully if not available.
        self._l2cs: object | None = None
        self._l2cs_enabled = False
        self._init_l2cs()

        # Blink hold state.
        self._last_valid_gaze: tuple[int, int] | None = None
        self._blink_start: float | None = None

        # eyetrax estimator — set externally by the orchestrator after calibration load.
        self.estimator: object | None = None
        self.is_calibrated: bool = False

    def _init_l2cs(self) -> None:
        """Load L2CS-Net ONNX model; disable cleanly if absent or broken."""
        mode = self._gaze_cfg.model_mode
        if mode == "mlp":
            logger.info("Gaze mode = mlp; L2CS-Net disabled.")
            return

        model_path = Paths.l2cs_model()
        if not model_path.exists():
            msg = "L2CS-Net model not found at %s. Run tools/download_l2cs.py."
            if self._gaze_cfg.strict_l2cs:
                raise FileNotFoundError(msg % model_path)
            logger.warning(msg + " Falling back to MLP only.", model_path)
            return

        try:
            from gazecontrol.gaze.l2cs_model import L2CSModel  # type: ignore[attr-defined]

            self._l2cs = L2CSModel(str(model_path))
            self._l2cs_enabled = getattr(self._l2cs, "is_loaded", False)
        except Exception:
            msg = "L2CS-Net failed to load."
            if self._gaze_cfg.strict_l2cs:
                raise
            logger.warning(msg + " Falling back to MLP only.", exc_info=True)

    def process(self, ctx: FrameContext) -> FrameContext:
        """Run the full gaze pipeline for one frame tick."""
        if not ctx.capture_ok or ctx.frame_bgr is None or ctx.frame_rgb is None:
            return ctx

        if not self.is_calibrated or self.estimator is None:
            return ctx

        t0 = ctx.t0

        try:
            # 1. Landmark extraction via eyetrax.
            features, blink = self.estimator.extract_features(ctx.frame_bgr)  # type: ignore[union-attr]
            ctx.blink = bool(blink)

            if blink:
                ctx = self._handle_blink(ctx, t0)
                return ctx

            self._blink_start = None

            if features is None or (ctx.quality is not None and not ctx.quality.is_usable):
                ctx.gaze_point = self._last_valid_gaze
                return ctx

            # 2. TinyMLP prediction → raw screen coords.
            raw = self.estimator.predict([features])[0]  # type: ignore[union-attr]
            px, py = float(raw[0]), float(raw[1])
            ctx.gaze_raw = (px, py)
            ctx.landmarks = features  # store for L2CS crop

            # 3. L2CS-Net ensemble (optional).
            if self._l2cs_enabled and self._l2cs is not None:
                px, py = self._apply_l2cs_ensemble(ctx, px, py)

            # 4. OneEuroFilter.
            fx = self._filter_x.filter(px, timestamp=t0)
            fy = self._filter_y.filter(py, timestamp=t0)
            ctx.gaze_filtered = (fx, fy)

            # 5. Drift correction.
            cx, cy = self._drift_corrector.correct(fx, fy)
            ctx.gaze_corrected = (cx, cy)

            # 6. Fixation detection.
            gaze_event = self._fixation_detector.update(cx, cy, t0)
            ctx.fixation_event = gaze_event

            # 7. Choose output point (centroid during fixation, raw during saccade).
            if gaze_event.type == "saccade":
                gaze_point = (int(cx), int(cy))
            elif gaze_event.centroid:
                gx, gy = gaze_event.centroid
                gaze_point = (int(gx), int(gy))
            else:
                gaze_point = (int(cx), int(cy))

            ctx.gaze_point = gaze_point
            self._last_valid_gaze = gaze_point

        except Exception:
            logger.warning("GazeStage: predict failed", exc_info=True)

        return ctx

    def _handle_blink(self, ctx: FrameContext, t0: float) -> FrameContext:
        """Hold last valid gaze during blink; reset filters on long blinks."""
        blink_hold_max_s = get_settings().intent.blink_hold_max_s
        if self._blink_start is None:
            self._blink_start = t0
        blink_duration = t0 - self._blink_start
        if blink_duration < blink_hold_max_s and self._last_valid_gaze:
            ctx.gaze_point = self._last_valid_gaze
        elif blink_duration >= blink_hold_max_s:
            self._filter_x.reset()
            self._filter_y.reset()
        return ctx

    def _apply_l2cs_ensemble(
        self, ctx: FrameContext, px: float, py: float
    ) -> tuple[float, float]:
        """Blend TinyMLP prediction with L2CS-Net via weighted ensemble."""
        # Use real landmarks for crop if available.
        landmarks_np: np.ndarray | None = None
        if ctx.landmarks is not None:
            raw_lm = getattr(ctx.landmarks, "face_landmarks", None)
            if raw_lm is not None:
                try:
                    landmarks_np = np.array(
                        [(lm.x, lm.y, lm.z) for lm in raw_lm], dtype=np.float32
                    )
                except Exception:
                    landmarks_np = None

        if landmarks_np is not None:
            face_crop = self._face_cropper.crop_from_landmarks(ctx.frame_bgr, landmarks_np)
        else:
            face_crop = self._face_cropper.crop_from_frame(ctx.frame_bgr)

        if face_crop is None:
            return px, py

        l2cs_angles = self._l2cs.predict(face_crop)  # type: ignore[union-attr]
        if l2cs_angles is None:
            return px, py

        yaw, pitch = l2cs_angles
        l2cs_xy = self._gaze_mapper.predict(yaw, pitch)
        if l2cs_xy is None:
            return px, py

        cfg = self._gaze_cfg
        # If GazeMapper is unfitted, it returns None (fixed); safe to blend.
        mlp_w = cfg.ensemble_weight_mlp
        l2cs_w = cfg.ensemble_weight_l2cs
        return (
            mlp_w * px + l2cs_w * l2cs_xy[0],
            mlp_w * py + l2cs_w * l2cs_xy[1],
        )

    def on_action(self, gaze_point: tuple[int, int], target_window: dict) -> None:
        """Notify drift corrector of a confirmed user action (implicit recal)."""
        self._drift_corrector.on_action(gaze_point, target_window)

    def reset_filters(self) -> None:
        """Reset OneEuro filters (e.g. after long pause)."""
        self._filter_x.reset()
        self._filter_y.reset()
