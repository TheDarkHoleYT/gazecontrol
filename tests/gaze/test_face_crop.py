"""Tests for FaceCropper — bounds, normalization."""
from __future__ import annotations

import numpy as np

from gazecontrol.gaze.face_crop import FaceCropper

# Build a minimal landmark array (478 × 3), all in the center.
_LANDMARKS_CENTER = np.full((478, 3), 0.5, dtype=np.float32)
# Spread some landmarks to give a meaningful bounding box.
_LANDMARKS_SPREAD = np.random.default_rng(0).uniform(0.3, 0.7, (478, 3)).astype(np.float32)


def _bgr_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestCropFromLandmarks:
    def test_returns_correct_shape(self):
        cropper = FaceCropper()
        frame = _bgr_frame()
        crop = cropper.crop_from_landmarks(frame, _LANDMARKS_SPREAD)
        assert crop is not None
        assert crop.shape == (1, 3, 224, 224)

    def test_returns_float32(self):
        cropper = FaceCropper()
        frame = _bgr_frame()
        crop = cropper.crop_from_landmarks(frame, _LANDMARKS_SPREAD)
        assert crop is not None
        assert crop.dtype == np.float32

    def test_imagenet_normalized_range(self):
        """Values should be roughly in [-3, 3] after ImageNet normalization."""
        cropper = FaceCropper()
        frame = _bgr_frame()
        crop = cropper.crop_from_landmarks(frame, _LANDMARKS_SPREAD)
        assert crop is not None
        assert crop.min() > -5.0
        assert crop.max() < 5.0


class TestCropFromFrame:
    def test_fallback_returns_crop(self):
        cropper = FaceCropper()
        frame = _bgr_frame()
        crop = cropper.crop_from_frame(frame)
        assert crop is not None
        assert crop.shape == (1, 3, 224, 224)

    def test_custom_rect_returns_crop(self):
        cropper = FaceCropper()
        frame = _bgr_frame()
        crop = cropper.crop_from_frame(frame, face_rect=(0.3, 0.2, 0.7, 0.8))
        assert crop is not None
        assert crop.shape == (1, 3, 224, 224)

    def test_degenerate_rect_returns_none(self):
        """When rect collapses to a point, crop_from_frame should return None."""
        cropper = FaceCropper()
        frame = _bgr_frame()
        # x_min == x_max → zero width.
        crop = cropper.crop_from_frame(frame, face_rect=(0.5, 0.5, 0.5, 0.5))
        assert crop is None
