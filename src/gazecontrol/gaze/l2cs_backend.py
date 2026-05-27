"""L2CSBackend — appearance-based gaze estimation via L2CS-Net (ONNX).

Pipeline per frame:
    1. Crop the face from the BGR frame using MediaPipe Face Detection
       (BlazeFace) — light-weight, ~3 ms.
    2. Normalize the crop (ImageNet mean/std) → tensor (1, 3, 224, 224).
    3. Run L2CS-Net ONNX → (yaw, pitch) angles in degrees.
    4. Map angles → screen pixels via :class:`GazeMapper` (loaded from the
       calibration profile).

Resource ownership: the ONNX session and Face Detection model are
allocated in ``start()`` and released in ``stop()``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from gazecontrol.errors import ModelLoadError
from gazecontrol.gaze.backend import GazePrediction, GazeQuality
from gazecontrol.gaze.blink import BlinkDetector, mean_ear
from gazecontrol.gaze.confidence import (
    AngleJitter,
    confidence_score,
    laplacian_variance,
)
from gazecontrol.gaze.face_cascade import (
    FaceDetectionCascade,
    bbox_from_landmarks_norm,
)
from gazecontrol.gaze.face_crop import FaceCropper
from gazecontrol.gaze.face_tracking import FaceTracker, NormalisedBBox
from gazecontrol.gaze.gaze_mapper import GazeMapper
from gazecontrol.gaze.head_pose import solve_head_pose
from gazecontrol.paths import Paths
from gazecontrol.settings import ConfidenceModelSettings

logger = logging.getLogger(__name__)


class L2CSBackend:
    """Gaze backend wrapping L2CS-Net + GazeMapper."""

    name = "l2cs"

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        profile_name: str = "default",
        strict: bool = False,
        *,
        face_lock_iou_threshold: float = 0.3,
        confidence_model: ConfidenceModelSettings | None = None,
        mapper_type: str = "poly_ridge",
        enable_face_landmarker: bool = True,
        blink_closed_threshold: float = 0.18,
        blink_open_margin: float = 0.04,
        blink_min_closed_frames: int = 2,
        max_replay_frames: int = 5,
    ) -> None:
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._profile_name = profile_name
        self._strict = strict
        self._mapper_type = mapper_type
        self._enable_face_landmarker = bool(enable_face_landmarker)

        self._model: Any = None
        self._face_cropper: FaceCropper | None = None
        self._mapper: GazeMapper | None = None
        self._face_detector: Any = None
        self._face_landmarker: Any = None
        self._face_tracker = FaceTracker(lock_iou_threshold=face_lock_iou_threshold)
        self._conf_cfg = confidence_model or ConfidenceModelSettings()
        self._angle_jitter = AngleJitter(
            window=self._conf_cfg.jitter_window,
            jitter_saturation_deg=self._conf_cfg.jitter_saturation_deg,
        )
        self._last_face_score: float = 0.5
        self._blink_detector = BlinkDetector(
            closed_threshold=blink_closed_threshold,
            open_margin=blink_open_margin,
            min_closed_frames=blink_min_closed_frames,
        )
        self._face_cascade = FaceDetectionCascade(max_replay_frames=max_replay_frames)
        # Stash of normalised face landmarks from the current frame so
        # both _detect_face (cascade fallback) and the head-pose / blink
        # path can use them without re-running the landmarker.
        self._cached_landmarks_norm: dict[int, tuple[float, float]] | None = None

    def start(self) -> bool:
        """Load the ONNX model, the gaze mapper, and the face detector."""
        from gazecontrol.gaze.l2cs_model import L2CSModel

        model_path = Paths.l2cs_model()
        if not model_path.exists():
            msg = f"L2CS model not found at {model_path}"
            if self._strict:
                raise ModelLoadError(msg)
            logger.warning("%s — L2CSBackend disabled.", msg)
            return False

        try:
            self._model = L2CSModel(str(model_path))
        except (OSError, RuntimeError, ValueError):
            logger.exception("L2CSBackend: failed to load ONNX model.")
            if self._strict:
                raise
            return False

        self._face_cropper = FaceCropper()
        self._mapper = GazeMapper(
            screen_w=self._screen_w,
            screen_h=self._screen_h,
            mapper_type=self._mapper_type,
        )

        profile_path = Paths.gaze_profile(self._profile_name)
        if profile_path.exists():
            if not self._mapper.load(profile_path):
                logger.warning("L2CSBackend: failed to load profile %s.", profile_path)
        else:
            logger.warning(
                "L2CSBackend: gaze profile %s not found; run --calibrate-gaze.",
                profile_path,
            )

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            blaze_path = Paths.blaze_face_model()
            if not blaze_path.exists():
                logger.warning(
                    "L2CSBackend: BlazeFace model not found at %s; "
                    "face detection disabled (will use centre-frame fallback).",
                    blaze_path,
                )
                self._face_detector = None
            else:
                options = mp_vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(blaze_path)),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    min_detection_confidence=0.5,
                )
                self._face_detector = mp_vision.FaceDetector.create_from_options(options)
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError):
            logger.exception("L2CSBackend: MediaPipe FaceDetector unavailable.")
            self._face_detector = None

        # --- G2 + G6: Face Landmarker for head-pose PnP + EAR blink -----
        if self._enable_face_landmarker:
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision

                lm_path = Paths.face_landmarker()
                if not lm_path.exists():
                    logger.warning(
                        "L2CSBackend: Face Landmarker model not found at %s; "
                        "head-pose PnP and EAR blink detection disabled.",
                        lm_path,
                    )
                    self._face_landmarker = None
                else:
                    lm_options = mp_vision.FaceLandmarkerOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(lm_path)),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_faces=1,
                        # Output blendshapes / facial transformation matrices
                        # are off by default — we only need the landmarks.
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                    )
                    self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(
                        lm_options
                    )
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError):
                logger.exception(
                    "L2CSBackend: MediaPipe FaceLandmarker unavailable — "
                    "head-pose / EAR blink disabled."
                )
                self._face_landmarker = None
        return True

    def stop(self) -> None:
        """Release the ONNX session and MediaPipe face detector / landmarker."""
        if self._face_detector is not None:
            try:
                self._face_detector.close()
            except (RuntimeError, OSError):
                logger.debug("L2CSBackend: face detector close failed.", exc_info=True)
            self._face_detector = None
        if self._face_landmarker is not None:
            try:
                self._face_landmarker.close()
            except (RuntimeError, OSError):
                logger.debug("L2CSBackend: face landmarker close failed.", exc_info=True)
            self._face_landmarker = None
        self._model = None
        self._face_cropper = None
        self._mapper = None
        self._angle_jitter.reset()
        self._face_tracker.reset()
        self._blink_detector.reset()
        self._face_cascade.reset()
        self._cached_landmarks_norm = None
        self._last_face_score = 0.5

    def is_calibrated(self) -> bool:
        """True when both the model and the gaze mapper are usable."""
        return self._model is not None and self._mapper is not None and self._mapper.is_fitted

    def predict(
        self,
        frame_bgr: np.ndarray[Any, Any],
        frame_rgb: np.ndarray[Any, Any],
        timestamp: float,
    ) -> GazePrediction | None:
        """Run face crop → L2CS → mapper → screen coordinate."""
        if self._model is None or self._face_cropper is None or self._mapper is None:
            return None
        if not self._mapper.is_fitted:
            return None

        # --- G2 + G4 + G6 — landmarks first, then cascade, then HP/blink ---
        # Running the Face Landmarker before the cascade lets the cascade
        # use its bounding box as a Tier 2 fallback when BlazeFace misses.
        self._run_face_landmarker(frame_rgb)
        landmarker_bbox = (
            bbox_from_landmarks_norm(self._cached_landmarks_norm)
            if self._cached_landmarks_norm
            else None
        )

        tracked = self._detect_face(frame_rgb)
        blaze_bbox = tracked[0] if tracked is not None else None
        face_id = tracked[1] if tracked is not None else None
        multi = bool(tracked[2]) if tracked is not None else False

        outcome = self._face_cascade.step(
            blaze_bbox=blaze_bbox,
            landmarker_bbox=landmarker_bbox,
        )
        if outcome is None:
            return None
        face_rect = outcome.bbox
        quality = GazeQuality.NONE
        if multi:
            quality |= GazeQuality.MULTI_FACE
        if outcome.tier != "blaze":
            # The primary detector missed — mark the frame as partially
            # occluded so the HUD / telemetry can react. We re-use the
            # OCCLUDED bit rather than introducing a new flag because
            # the downstream meaning is identical ("trust this sample less").
            quality |= GazeQuality.OCCLUDED

        crop = self._face_cropper.crop_from_frame(frame_bgr, face_rect=face_rect)
        if crop is None:
            return None

        head_pose_rad, is_blink = self._head_pose_blink_from_cached(
            (int(frame_rgb.shape[0]), int(frame_rgb.shape[1]))
        )
        if is_blink:
            quality |= GazeQuality.BLINK

        try:
            angles = self._model.predict(crop)
        except (RuntimeError, ValueError):
            logger.debug("L2CSBackend: ONNX predict failed.", exc_info=True)
            return None
        if angles is None:
            return None
        yaw, pitch = angles
        screen = self._mapper.predict(yaw, pitch, head_pose=head_pose_rad)
        if screen is None:
            return None
        # --- G1: per-frame confidence -----------------------------------
        # Push the new angle into the jitter buffer first so the score
        # reflects the current sample's contribution.
        self._angle_jitter.push(yaw, pitch)
        sharpness = laplacian_variance(crop)
        cfg = self._conf_cfg
        conf = confidence_score(
            face_score=self._last_face_score,
            sharpness=sharpness,
            jitter=self._angle_jitter.score(),
            sharpness_floor=cfg.sharpness_floor,
            sharpness_ceiling=cfg.sharpness_ceiling,
            w_face=cfg.w_face,
            w_sharpness=cfg.w_sharpness,
            w_jitter=cfg.w_jitter,
            bias=cfg.bias,
            floor=cfg.floor,
            ceiling=cfg.ceiling,
        )
        # During a confirmed blink, gaze direction is unreliable —
        # collapse confidence to zero so confidence-weighted fusion
        # (G3) ignores this sample on the consumer side.
        if is_blink:
            conf = 0.0
        return GazePrediction(
            screen_xy=(int(screen[0]), int(screen[1])),
            confidence=conf,
            yaw_pitch_deg=(yaw, pitch),
            blink=is_blink,
            head_pose_rad=head_pose_rad,
            backend_name=self.name,
            face_bbox_norm=face_rect,
            face_id=face_id,
            quality_flags=int(quality),
        )

    def _run_face_landmarker(
        self,
        frame_rgb: np.ndarray[Any, Any],
    ) -> None:
        """Run Face Landmarker once and cache the result for later helpers.

        Populates :attr:`_cached_landmarks_norm` with a
        ``{landmark_id → (x_norm, y_norm)}`` dict (or ``None`` when the
        landmarker is disabled / missed). Caching avoids running the
        landmarker twice per frame (once for the cascade Tier 2 fallback,
        once for head-pose / blink).
        """
        self._cached_landmarks_norm = None
        if self._face_landmarker is None:
            return
        try:
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._face_landmarker.detect(mp_image)
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.debug("L2CSBackend: face landmarker failed: %s", exc)
            return
        face_landmarks_list = getattr(result, "face_landmarks", None)
        if not face_landmarks_list:
            return
        landmarks = face_landmarks_list[0]
        self._cached_landmarks_norm = {
            i: (float(p.x), float(p.y)) for i, p in enumerate(landmarks)
        }

    def _head_pose_blink_from_cached(
        self,
        frame_shape: tuple[int, int],
    ) -> tuple[tuple[float, float, float] | None, bool]:
        """Derive ``(head_pose_rad, is_blinking)`` from the cached landmarks.

        ``frame_shape`` is ``(height, width)`` matching ``np.ndarray.shape[:2]``.
        Returns ``(None, blink_decay)`` when the cache is empty so the
        blink detector's hysteresis streak decays naturally instead of
        latching on a stale ``True``.
        """
        if not self._cached_landmarks_norm:
            return None, self._blink_detector.update(None)
        h, w = frame_shape[0], frame_shape[1]
        if w <= 0 or h <= 0:
            return None, self._blink_detector.update(None)
        landmarks_px: dict[int, tuple[float, float]] = {
            i: (xy[0] * w, xy[1] * h)
            for i, xy in self._cached_landmarks_norm.items()
        }
        head_pose = solve_head_pose(landmarks_px, (w, h))
        ear = mean_ear(landmarks_px)
        is_blinking = self._blink_detector.update(ear)
        return head_pose, is_blinking

    def _detect_face(
        self,
        frame_rgb: np.ndarray[Any, Any],
    ) -> tuple[NormalisedBBox, int, bool] | None:
        """Run BlazeFace and pick the tracked face via :class:`FaceTracker`.

        Returns the winning ``(bbox_norm, face_id, multi_face)`` triple or
        ``None`` when no face was detected (or the detector is disabled).
        ``multi_face`` is True when at least two candidates were seen,
        regardless of whether the lock changed — callers OR
        :data:`GazeQuality.MULTI_FACE` into the prediction's quality
        flags so the HUD can warn the user.
        """
        if self._face_detector is None:
            return None
        try:
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._face_detector.detect(mp_image)
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.debug("L2CSBackend: face detect failed: %s", exc)
            return None
        detections = getattr(result, "detections", None)
        if not detections:
            return None
        h, w = frame_rgb.shape[:2]
        if w <= 0 or h <= 0:
            return None
        # Collect (bbox_norm, score) tuples for the FaceTracker.
        candidates: list[tuple[NormalisedBBox, float]] = []
        for det in detections:
            bb = det.bounding_box
            x_min = max(0.0, bb.origin_x / w)
            y_min = max(0.0, bb.origin_y / h)
            x_max = min(1.0, (bb.origin_x + bb.width) / w)
            y_max = min(1.0, (bb.origin_y + bb.height) / h)
            if x_max <= x_min or y_max <= y_min:
                continue
            # MediaPipe Tasks detections expose categories[0].score as the
            # confidence; treat missing as 0.5 to keep the tie-break neutral.
            score = 0.5
            cats = getattr(det, "categories", None)
            if cats:
                cat_score = getattr(cats[0], "score", None)
                if cat_score is not None:
                    score = float(cat_score)
            candidates.append(((x_min, y_min, x_max, y_max), score))
        tracked = self._face_tracker.update(candidates)
        # Cache the winning detection's score so predict() can feed it
        # into the per-frame confidence model (G1).
        if tracked is not None:
            winner_bbox = tracked[0]
            for bbox, score in candidates:
                if bbox == winner_bbox:
                    self._last_face_score = score
                    break
        else:
            self._last_face_score = 0.0
        return tracked
