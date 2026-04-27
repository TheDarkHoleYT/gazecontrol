"""FramePreprocessor — optional frame enhancement for hand-gesture detection.

Operations (all optional, controlled by constructor flags):

1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization): normalises
   illumination by working on the L channel of LAB.  Helps MediaPipe
   HandLandmarker detect hands under uneven or dim lighting.
2. **Sharpening**: mild unsharp-mask 3×3 kernel blended 70/30.  Improves
   landmark precision on slightly soft webcam frames.
3. **Quality score**: Laplacian variance on a ½-scale grey image (≈3 ms at
   1280×720).  Frames below the blur threshold are flagged as not usable
   so downstream stages can skip them.

Usage::

    preprocessor = FramePreprocessor()
    enhanced_bgr, quality = preprocessor.process(frame_bgr)
    if quality.is_usable:
        # Pass enhanced_bgr to hand detector.
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class FrameQuality:
    """Per-frame quality metrics from the preprocessor."""

    laplacian_var: float  # Laplacian variance (> 80 = sharp, < 20 = blurry).
    brightness_mean: float  # Mean pixel brightness [0, 255].
    is_usable: bool  # True when frame passes all quality thresholds.


class FramePreprocessor:
    """Optionally enhance BGR webcam frames for MediaPipe hand detection.

    Args:
        clahe_clip_limit: CLAHE amplification limit (2.0–4.0).
        clahe_tile_grid:  CLAHE tile grid size in pixels (8×8 typical).
        sharpen:          Apply mild unsharp-mask sharpening kernel.
        blur_threshold:   Minimum Laplacian variance for the frame to be
                          considered sharp.  Set to ``0.0`` to disable.
        brightness_min:   Minimum acceptable mean brightness.
        brightness_max:   Maximum acceptable mean brightness.
    """

    # Mild unsharp-mask 3×3 kernel.
    _SHARPEN_KERNEL = np.array(
        [
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple[int, int] = (8, 8),
        sharpen: bool = True,
        blur_threshold: float = 15.0,
        brightness_min: float = 30.0,
        brightness_max: float = 230.0,
    ) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid,
        )
        self._sharpen = sharpen
        self._blur_thr = blur_threshold
        self._bri_min = brightness_min
        self._bri_max = brightness_max

    def process(self, frame_bgr: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], FrameQuality]:
        """Process one BGR frame.

        Args:
            frame_bgr: Input frame in BGR format.

        Returns:
            ``(enhanced_bgr, quality)`` — enhanced frame and quality metrics.
        """
        quality = self._compute_quality(frame_bgr)

        # CLAHE on L channel of LAB (preserves hue/saturation).
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a, b])
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # Mild sharpening: 70% sharpened + 30% original to avoid artefacts.
        if self._sharpen:
            sharpened = cv2.filter2D(enhanced, -1, self._SHARPEN_KERNEL)
            enhanced = cv2.addWeighted(sharpened, 0.7, enhanced, 0.3, 0)

        return enhanced, quality

    def compute_quality(self, frame_bgr: np.ndarray[Any, Any]) -> FrameQuality:
        """Compute quality metrics without applying any enhancement.

        Useful when enhancement is disabled but the quality gate is still needed.
        """
        return self._compute_quality(frame_bgr)

    def _compute_quality(self, frame_bgr: np.ndarray[Any, Any]) -> FrameQuality:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Half-scale before Laplacian saves ~3 ms at 1280×720.
        small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
        lap_var = float(cv2.Laplacian(small, cv2.CV_64F).var())
        brightness = float(gray.mean())
        is_usable = lap_var >= self._blur_thr and self._bri_min <= brightness <= self._bri_max
        return FrameQuality(
            laplacian_var=lap_var,
            brightness_mean=brightness,
            is_usable=is_usable,
        )
