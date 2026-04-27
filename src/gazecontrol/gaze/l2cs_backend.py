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
            import mediapipe as mp

            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5,
            )
        except (ImportError, RuntimeError, OSError):
            logger.exception("L2CSBackend: MediaPipe FaceDetection unavailable.")
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
        """Return the largest detected face rect in normalised coords, or None."""
        if self._face_detector is None:
            return None
        try:
            result = self._face_detector.process(frame_rgb)
        except (RuntimeError, ValueError) as exc:
            logger.debug("L2CSBackend: face detect failed: %s", exc)
            return None
        if not getattr(result, "detections", None):
            return None
        best = max(
            result.detections,
            key=lambda d: (
                d.location_data.relative_bounding_box.width
                * d.location_data.relative_bounding_box.height
            ),
        )
        bb = best.location_data.relative_bounding_box
        x_min = max(0.0, bb.xmin)
        y_min = max(0.0, bb.ymin)
        x_max = min(1.0, bb.xmin + bb.width)
        y_max = min(1.0, bb.ymin + bb.height)
        return (x_min, y_min, x_max, y_max)
