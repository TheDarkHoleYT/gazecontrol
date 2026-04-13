"""
GazeMapper — converte (yaw, pitch) di L2CS-Net in coordinate pixel schermo.

Approccio:
  L2CS-Net produce angoli in gradi (yaw, pitch) nello spazio camera.
  Per mappare questi angoli sullo schermo usiamo un polynomial regression
  di grado 2 fittato durante la calibrazione (few-shot personalization).

  Senza calibrazione personale, usa una stima geometrica semplificata
  basata su distanza focale stimata e dimensioni schermo.

  La mapping personalizzata è molto più accurata (errore atteso < 1.5° vs ~4°
  del modello cross-person) e richiede solo 5-13 punti di calibrazione.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class GazeMapper:
    """
    Mappa angoli gaze (yaw, pitch) → coordinate schermo (px_x, px_y).

    Usa polynomial regression degree 2 con termini incrociati:
        features = [yaw, pitch, yaw², pitch², yaw*pitch, head_yaw, head_pitch, head_roll]
    """

    POLY_DEGREE = 2

    def __init__(self, screen_w: int = 1920, screen_h: int = 1080):
        self._sw = screen_w
        self._sh = screen_h
        self._coef_x: object | None = None  # Ridge estimator when fitted
        self._coef_y: object | None = None
        self._scaler: object | None = None   # StandardScaler when fitted (always init'd)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        gaze_angles: np.ndarray,     # (N, 2): (yaw, pitch) in gradi
        screen_points: np.ndarray,   # (N, 2): (px_x, px_y)
        head_poses: np.ndarray | None = None,  # (N, 3): (yaw, pitch, roll) in rad
    ) -> float:
        """
        Fitta la mapping sui dati di calibrazione.

        Returns:
            Errore medio di cross-validation (Leave-One-Out) in pixel.
        """
        X = self._build_features(gaze_angles, head_poses)
        y_x = screen_points[:, 0]
        y_y = screen_points[:, 1]

        # Ridge regression (robusto a pochi punti di dati)
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)

        reg_x = Ridge(alpha=1.0)
        reg_y = Ridge(alpha=1.0)
        reg_x.fit(Xs, y_x)
        reg_y.fit(Xs, y_y)

        self._coef_x = reg_x
        self._coef_y = reg_y
        self._is_fitted = True

        # Calcola LOO error
        errors = []
        n = len(X)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            Xs_tr = self._scaler.transform(X[mask])
            rx = Ridge(alpha=1.0).fit(Xs_tr, y_x[mask])
            ry = Ridge(alpha=1.0).fit(Xs_tr, y_y[mask])
            pred_x = rx.predict(self._scaler.transform(X[i:i+1]))[0]
            pred_y = ry.predict(self._scaler.transform(X[i:i+1]))[0]
            errors.append(np.hypot(pred_x - y_x[i], pred_y - y_y[i]))

        loo_error = float(np.mean(errors))
        logger.info("GazeMapper: LOO error = %.1f px (%.2f°)",
                    loo_error, loo_error / 44.0)
        return loo_error

    def predict(
        self,
        yaw: float,
        pitch: float,
        head_pose: tuple[float, float, float] | None = None,
    ) -> tuple[float, float] | None:
        """
        Predice le coordinate schermo da (yaw, pitch) in gradi.

        Returns:
            (px_x, px_y) oppure None se non ancora fittato.
        """
        if not self._is_fitted or self._scaler is None:
            # Not calibrated — return None so the ensemble can skip L2CS contribution.
            return None

        angles = np.array([[yaw, pitch]])
        hp = np.array([list(head_pose)]) if head_pose else None
        X = self._build_features(angles, hp)
        Xs = self._scaler.transform(X)

        px_x = float(self._coef_x.predict(Xs)[0])
        px_y = float(self._coef_y.predict(Xs)[0])

        # Clamp ai bordi schermo
        px_x = max(0.0, min(float(self._sw - 1), px_x))
        px_y = max(0.0, min(float(self._sh - 1), px_y))
        return px_x, px_y

    def save(self, path: str | Path) -> None:
        data = {
            'coef_x': self._coef_x,
            'coef_y': self._coef_y,
            'scaler': self._scaler if hasattr(self, '_scaler') else None,
            'screen': (self._sw, self._sh),
            'is_fitted': self._is_fitted,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str | Path) -> bool:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._coef_x = data['coef_x']
            self._coef_y = data['coef_y']
            self._scaler = data.get('scaler')
            self._sw, self._sh = data.get('screen', (1920, 1080))
            loaded_fitted = data.get('is_fitted', False)
            # If the saved scaler is None, the model cannot predict — treat as not fitted
            if loaded_fitted and self._scaler is None:
                logger.warning("Profile has is_fitted=True but scaler=None; resetting to not-fitted")
                self._is_fitted = False
            else:
                self._is_fitted = loaded_fitted
            return True
        except Exception:
            logger.exception("Errore caricamento GazeMapper da %s", path)
            return False

    # ------------------------------------------------------------------

    def _build_features(
        self,
        gaze_angles: np.ndarray,            # (N, 2)
        head_poses: np.ndarray | None,      # (N, 3) o None
    ) -> np.ndarray:
        yaw   = gaze_angles[:, 0:1]
        pitch = gaze_angles[:, 1:2]
        feats = [yaw, pitch, yaw**2, pitch**2, yaw * pitch]
        if head_poses is not None:
            feats.append(head_poses)
        return np.hstack(feats)

    def _geometric_estimate(self, yaw: float, pitch: float) -> tuple[float, float]:
        """
        Stima geometrica naive senza calibrazione personale.
        Utile solo come fallback iniziale; errore atteso ~4-6°.
        """
        import math
        # Distanza focale stimata: 60cm, FOV camera ~60°
        focal_px = self._sw / (2 * math.tan(math.radians(30)))

        px_x = self._sw / 2 - math.tan(math.radians(yaw)) * focal_px
        px_y = self._sh / 2 - math.tan(math.radians(pitch)) * focal_px

        px_x = max(0.0, min(float(self._sw - 1), px_x))
        px_y = max(0.0, min(float(self._sh - 1), px_y))
        return px_x, px_y
