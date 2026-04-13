"""
FaceCropper — estrae un crop del volto 224×224 per L2CS-Net.

Usa MediaPipe Face Detection (leggero, non il Face Landmarker pesante) per
localizzare il volto, poi ritaglia e normalizza secondo i parametri di training
di L2CS-Net (ResNet-50, ImageNet mean/std).

Ottimizzazione: usa il Face Mesh già estratto da EyeTrax (landmarks) per
derivare il bounding box senza rieseguire una seconda rete di face detection.
Se i landmarks non sono disponibili, fa fallback su una stima basata su
coordinate di landmark hardcoded.
"""
from __future__ import annotations

import cv2
import numpy as np


# ImageNet normalization (usata da L2CS-Net/ResNet-50)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Padding attorno al bounding box del volto (% della dimensione)
_FACE_PAD = 0.4


class FaceCropper:
    """
    Ritaglia e normalizza il volto da un frame BGR per input a L2CS-Net.

    Utilizzo con landmarks MediaPipe (metodo preferito):
        crop = cropper.crop_from_landmarks(frame_bgr, face_landmarks_478)

    Utilizzo standalone (solo per test, meno robusto):
        crop = cropper.crop_from_frame(frame_bgr)
    """

    TARGET_SIZE = (224, 224)

    def crop_from_landmarks(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray | None:
        """
        Ritaglia il volto usando landmarks MediaPipe (478 punti, coordinate norm.).

        Args:
            frame_bgr  : frame originale BGR.
            landmarks  : array (478, 3) con coordinate normalizzate [0, 1].

        Returns:
            Tensor numpy (1, 3, 224, 224) float32 pronto per ONNX, oppure None.
        """
        h, w = frame_bgr.shape[:2]

        # Usa subset di landmarks per bounding box stabile (contorno volto)
        face_contour = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                        162, 21, 54, 103, 67, 109]

        pts = landmarks[face_contour, :2]  # (N, 2) normalized
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        return self._crop_and_normalize(frame_bgr, x_min, y_min, x_max, y_max, w, h)

    def crop_from_frame(
        self,
        frame_bgr: np.ndarray,
        face_rect: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray | None:
        """
        Ritaglia il volto da un rettangolo normalizzato oppure assume centro frame.

        Args:
            frame_bgr : frame BGR.
            face_rect : (x_min, y_min, x_max, y_max) normalizzati [0,1], o None.

        Returns:
            Tensor numpy (1, 3, 224, 224) float32 o None.
        """
        h, w = frame_bgr.shape[:2]
        if face_rect is None:
            # Fallback: usa il 60% centrale del frame
            x_min, y_min, x_max, y_max = 0.2, 0.1, 0.8, 0.9
        else:
            x_min, y_min, x_max, y_max = face_rect

        return self._crop_and_normalize(frame_bgr, x_min, y_min, x_max, y_max, w, h)

    def _crop_and_normalize(
        self,
        frame_bgr: np.ndarray,
        x_min: float, y_min: float,
        x_max: float, y_max: float,
        w: int, h: int,
    ) -> np.ndarray | None:
        """Applica padding, ritaglia, ridimensiona e normalizza il crop."""
        # Padding proporzionale
        pad_x = (x_max - x_min) * _FACE_PAD
        pad_y = (y_max - y_min) * _FACE_PAD
        x0 = max(0.0, x_min - pad_x)
        y0 = max(0.0, y_min - pad_y)
        x1 = min(1.0, x_max + pad_x)
        y1 = min(1.0, y_max + pad_y)

        # Converti in pixel interi
        px0, py0 = int(x0 * w), int(y0 * h)
        px1, py1 = int(x1 * w), int(y1 * h)

        if px1 <= px0 or py1 <= py0:
            return None

        crop = frame_bgr[py0:py1, px0:px1]
        if crop.size == 0:
            return None

        # Ridimensiona a 224x224 con interpolazione bicubica
        crop = cv2.resize(crop, self.TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

        # BGR → RGB → float [0,1]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ImageNet normalization
        crop_norm = (crop_rgb - _IMAGENET_MEAN) / _IMAGENET_STD

        # HWC → CHW, aggiungi batch dim: (1, 3, 224, 224)
        crop_chw = crop_norm.transpose(2, 0, 1)
        return np.expand_dims(crop_chw, axis=0).astype(np.float32)
