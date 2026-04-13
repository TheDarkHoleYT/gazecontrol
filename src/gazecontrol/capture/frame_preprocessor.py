"""
FramePreprocessor — preprocessing enterprise dei frame webcam.

Operazioni:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization): normalizza l'illuminazione
   lavorando su canali separati. Cruciale per eye tracking sotto luci variabili.
2. Sharpening leggero: migliora i dettagli dell'iride per i landmark mediapipe.
3. Quality Score: stima la nitidezza tramite varianza del Laplaciano.
   Frame sfocati/in movimento vengono segnalati e opzionalmente scartati.

Utilizzo:
    preprocessor = FramePreprocessor()
    enhanced_frame, quality = preprocessor.process(frame_bgr)
    if quality.is_usable:
        # usa enhanced_frame per eye tracking
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FrameQuality:
    laplacian_var: float   # varianza Laplaciano (>80 = nitido, <20 = sfocato)
    brightness_mean: float # luminosità media [0, 255]
    is_usable: bool        # True se il frame è di qualità sufficiente


class FramePreprocessor:
    """
    Preprocessa i frame BGR per migliorare la qualità dell'eye tracking.

    Args:
        clahe_clip_limit    : limite di amplificazione CLAHE (2.0–4.0).
        clahe_tile_grid     : dimensione griglia CLAHE in pixel (8x8 tipico).
        sharpen             : applica kernel di sharpening.
        blur_threshold      : sotto questa var. Laplaciana il frame è considerato sfocato.
        brightness_min      : luminosità minima accettabile.
        brightness_max      : luminosità massima accettabile.
    """

    # Kernel di sharpening leggero (unsharp mask 3x3)
    _SHARPEN_KERNEL = np.array([
        [0,  -0.5,  0],
        [-0.5, 3,  -0.5],
        [0,  -0.5,  0],
    ], dtype=np.float32)

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple[int, int] = (8, 8),
        sharpen: bool = True,
        blur_threshold: float = 30.0,
        brightness_min: float = 30.0,
        brightness_max: float = 230.0,
    ):
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid,
        )
        self._sharpen = sharpen
        self._blur_thr = blur_threshold
        self._bri_min = brightness_min
        self._bri_max = brightness_max

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, FrameQuality]:
        """
        Processa un frame BGR.

        Returns:
            (enhanced_bgr, quality) — il frame migliorato e le sue metriche.
        """
        # Calcola quality sul frame originale (prima di alterarlo)
        quality = self._compute_quality(frame_bgr)

        # 1. Converti in LAB per applicare CLAHE solo al canale L (luminosità)
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # 2. Sharpening leggero
        if self._sharpen:
            sharpened = cv2.filter2D(enhanced, -1, self._SHARPEN_KERNEL)
            # Blend leggero: 70% sharpened, 30% original per evitare artefatti
            enhanced = cv2.addWeighted(sharpened, 0.7, enhanced, 0.3, 0)

        return enhanced, quality

    def _compute_quality(self, frame_bgr: np.ndarray) -> FrameQuality:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        is_usable = (
            lap_var >= self._blur_thr
            and self._bri_min <= brightness <= self._bri_max
        )
        return FrameQuality(
            laplacian_var=lap_var,
            brightness_mean=brightness,
            is_usable=is_usable,
        )
