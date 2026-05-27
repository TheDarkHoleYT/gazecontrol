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
from gazecontrol.gaze.backend import GazePrediction
from gazecontrol.gaze.face_crop import FaceCropper
from gazecontrol.gaze.gaze_mapper import GazeMapper
from gazecontrol.paths import Paths

logger = logging.getLogger(__name__)

_L2CS_CONFIDENCE = 0.8


class L2CSBackend:
    """Gaze backend wrapping L2CS-Net + GazeMapper."""

    name = "l2cs"

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        profile_name: str = "default",
        strict: bool = False,
    ) -> None:
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._profile_name = profile_name
        self._strict = strict

        self._model: Any = None
        self._face_cropper: FaceCropper | None = None
        self._mapper: GazeMapper | None = None
        self._face_detector: Any = None

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
        self._mapper = GazeMapper(screen_w=self._screen_w, screen_h=self._screen_h)

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
        return True

    def stop(self) -> None:
        """Release the ONNX session and MediaPipe face detector."""
        if self._face_detector is not None:
            try:
                self._face_detector.close()
            except (RuntimeError, OSError):
                logger.debug("L2CSBackend: face detector close failed.", exc_info=True)
            self._face_detector = None
        self._model = None
        self._face_cropper = None
        self._mapper = None

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

        face_rect = self._detect_face(frame_rgb)
        crop = self._face_cropper.crop_from_frame(frame_bgr, face_rect=face_rect)
        if crop is None:
            return None

        try:
            angles = self._model.predict(crop)
        except (RuntimeError, ValueError):
            logger.debug("L2CSBackend: ONNX predict failed.", exc_info=True)
            return None
        if angles is None:
            return None
        yaw, pitch = angles
        screen = self._mapper.predict(yaw, pitch)
        if screen is None:
            return None
        return GazePrediction(
            screen_xy=(int(screen[0]), int(screen[1])),
            confidence=_L2CS_CONFIDENCE,
            yaw_pitch_deg=(yaw, pitch),
            blink=False,
            backend_name=self.name,
        )

    def _detect_face(
        self,
        frame_rgb: np.ndarray[Any, Any],
    ) -> tuple[float, float, float, float] | None:
        """Return the largest detected face rect in normalised coords, or None.

        Uses the MediaPipe Tasks ``FaceDetector`` API. Result bounding boxes
        are in pixel coordinates; we normalise against the frame dimensions
        so downstream ``FaceCropper`` can pad and crop consistently.
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
        best = max(
            detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height,
        )
        bb = best.bounding_box
        h, w = frame_rgb.shape[:2]
        if w <= 0 or h <= 0:
            return None
        x_min = max(0.0, bb.origin_x / w)
        y_min = max(0.0, bb.origin_y / h)
        x_max = min(1.0, (bb.origin_x + bb.width) / w)
        y_max = min(1.0, (bb.origin_y + bb.height) / h)
        if x_max <= x_min or y_max <= y_min:
            return None
        return (x_min, y_min, x_max, y_max)
