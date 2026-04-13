"""GazeMapper — convert L2CS-Net (yaw, pitch) angles to screen pixel coordinates.

Persistence format: ``.npz`` + ``.meta.json`` (version-stable; no sklearn pickle).
Backward-compatible migration from old ``.pkl`` files is provided by
:func:`load_legacy_pkl`.

Predict contract:
- Returns ``(px_x, px_y)`` when fitted.
- Returns ``None`` when unfitted — callers must handle this explicitly.
  (Previously returned a noisy geometric estimate without warning.)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_FORMAT_VERSION = "1"


class GazeMapper:
    """Map gaze angles (yaw, pitch) → screen coordinates (px_x, px_y).

    Uses polynomial ridge regression degree 2::

        features = [yaw, pitch, yaw², pitch², yaw*pitch, (head_yaw, head_pitch, head_roll)?]

    Args:
        screen_w: Screen width in pixels.
        screen_h: Screen height in pixels.
    """

    POLY_DEGREE = 2

    def __init__(self, screen_w: int = 1920, screen_h: int = 1080) -> None:
        self._sw = screen_w
        self._sh = screen_h
        # Coefficient arrays (saved/loaded as npz arrays).
        self._coef_x: np.ndarray | None = None
        self._intercept_x: float = 0.0
        self._coef_y: np.ndarray | None = None
        self._intercept_y: float = 0.0
        # Scaler parameters (saved as npz arrays).
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True when the mapper has been trained and can predict."""
        return self._is_fitted

    def fit(
        self,
        gaze_angles: np.ndarray,
        screen_points: np.ndarray,
        head_poses: np.ndarray | None = None,
    ) -> float:
        """Fit the mapper on calibration data.

        Args:
            gaze_angles:   (N, 2) array of (yaw, pitch) in degrees.
            screen_points: (N, 2) array of (px_x, px_y) ground-truth screen coords.
            head_poses:    (N, 3) optional head pose (yaw, pitch, roll) in radians.

        Returns:
            Leave-one-out cross-validation error in pixels.
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X = self._build_features(gaze_angles, head_poses)
        y_x = screen_points[:, 0]
        y_y = screen_points[:, 1]

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        reg_x = Ridge(alpha=1.0)
        reg_y = Ridge(alpha=1.0)
        reg_x.fit(Xs, y_x)
        reg_y.fit(Xs, y_y)

        # Store coefficients as plain arrays (not estimators).
        self._coef_x = reg_x.coef_.copy()
        self._intercept_x = float(reg_x.intercept_)
        self._coef_y = reg_y.coef_.copy()
        self._intercept_y = float(reg_y.intercept_)
        self._scaler_mean = scaler.mean_.copy()
        self._scaler_scale = scaler.scale_.copy()
        self._is_fitted = True

        # Leave-one-out error.
        loo_error = self._loo_error(X, y_x, y_y)
        logger.info(
            "GazeMapper fitted: LOO error = %.1f px (%.2f°)", loo_error, loo_error / 44.0
        )
        return loo_error

    def predict(
        self,
        yaw: float,
        pitch: float,
        head_pose: tuple[float, float, float] | None = None,
    ) -> tuple[float, float] | None:
        """Predict screen coordinates from (yaw, pitch) in degrees.

        Returns:
            ``(px_x, px_y)`` clamped to screen bounds, or ``None`` if not fitted.
        """
        if not self._is_fitted or self._scaler_mean is None:
            return None

        angles = np.array([[yaw, pitch]])
        hp = np.array([list(head_pose)]) if head_pose else None
        X = self._build_features(angles, hp)
        Xs = (X - self._scaler_mean) / self._scaler_scale

        px_x = float(Xs @ self._coef_x + self._intercept_x)  # type: ignore[operator]
        px_y = float(Xs @ self._coef_y + self._intercept_y)  # type: ignore[operator]

        px_x = max(0.0, min(float(self._sw - 1), px_x))
        px_y = max(0.0, min(float(self._sh - 1), px_y))
        return px_x, px_y

    # ------------------------------------------------------------------
    # Persistence — npz + meta.json (version-stable)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save mapper to *path*.npz + *path*.meta.json.

        Example::

            mapper.save("profiles/default/gaze_mapper")
            # creates: gaze_mapper.npz, gaze_mapper.meta.json
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            str(path),
            coef_x=self._coef_x if self._coef_x is not None else np.array([]),
            coef_y=self._coef_y if self._coef_y is not None else np.array([]),
            intercept_x=np.array([self._intercept_x]),
            intercept_y=np.array([self._intercept_y]),
            scaler_mean=self._scaler_mean if self._scaler_mean is not None else np.array([]),
            scaler_scale=self._scaler_scale if self._scaler_scale is not None else np.array([]),
        )

        meta = {
            "format_version": _FORMAT_VERSION,
            "screen_w": self._sw,
            "screen_h": self._sh,
            "is_fitted": self._is_fitted,
        }
        meta_path = path.parent / (path.stem + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("GazeMapper saved to %s", path)

    def load(self, path: str | Path) -> bool:
        """Load mapper from *path*.npz + optional *path*.meta.json.

        Returns:
            True on success, False on any error.
        """
        path = Path(path)
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")

        try:
            data = np.load(str(npz_path), allow_pickle=False)
        except Exception:
            logger.exception("GazeMapper: failed to load npz from %s", npz_path)
            return False

        meta_path = npz_path.parent / (npz_path.stem + ".meta.json")
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("GazeMapper: could not read meta from %s", meta_path)

        try:
            self._coef_x = data["coef_x"]
            self._intercept_x = float(data["intercept_x"][0])
            self._coef_y = data["coef_y"]
            self._intercept_y = float(data["intercept_y"][0])
            self._scaler_mean = data["scaler_mean"]
            self._scaler_scale = data["scaler_scale"]
            self._sw, self._sh = (
                meta.get("screen_w", self._sw),
                meta.get("screen_h", self._sh),
            )
            fitted = meta.get("is_fitted", False)
            # Guard: arrays must be non-empty for predict to work.
            if fitted and (
                self._coef_x is None
                or self._coef_x.size == 0
                or self._scaler_mean is None
                or self._scaler_mean.size == 0
            ):
                logger.warning("GazeMapper: fitted=True but arrays are empty; resetting.")
                self._is_fitted = False
            else:
                self._is_fitted = fitted
            logger.info("GazeMapper loaded from %s (fitted=%s)", npz_path, self._is_fitted)
            return True
        except Exception:
            logger.exception("GazeMapper: error parsing npz data from %s", npz_path)
            return False

    # ------------------------------------------------------------------
    # Legacy migration
    # ------------------------------------------------------------------

    def load_legacy_pkl(self, pkl_path: str | Path) -> bool:
        """Migrate an old pickle-based profile to the new npz format.

        Loads the pickle file, extracts sklearn estimator coefficients,
        and populates the npz-backed fields.  Does NOT save automatically
        — call :meth:`save` afterwards.

        Returns:
            True if migration succeeded, False otherwise.
        """
        import pickle  # noqa: S403 — only used for migration

        pkl_path = Path(pkl_path)
        try:
            with pkl_path.open("rb") as fh:
                data = pickle.load(fh)  # noqa: S301
        except Exception:
            logger.exception("GazeMapper: could not load legacy pkl %s", pkl_path)
            return False

        try:
            reg_x = data.get("coef_x")
            reg_y = data.get("coef_y")
            scaler = data.get("scaler")

            if reg_x is None or reg_y is None or scaler is None:
                logger.warning("GazeMapper: legacy pkl missing required keys.")
                return False

            # Ridge estimator → plain arrays.
            if hasattr(reg_x, "coef_"):
                self._coef_x = reg_x.coef_.copy()
                self._intercept_x = float(reg_x.intercept_)
            else:
                self._coef_x = np.asarray(reg_x, dtype=np.float64)
                self._intercept_x = 0.0

            if hasattr(reg_y, "coef_"):
                self._coef_y = reg_y.coef_.copy()
                self._intercept_y = float(reg_y.intercept_)
            else:
                self._coef_y = np.asarray(reg_y, dtype=np.float64)
                self._intercept_y = 0.0

            if hasattr(scaler, "mean_"):
                self._scaler_mean = scaler.mean_.copy()
                self._scaler_scale = scaler.scale_.copy()
            else:
                logger.warning("GazeMapper: legacy scaler has no mean_/scale_; cannot migrate.")
                return False

            sw, sh = data.get("screen", (1920, 1080))
            self._sw, self._sh = int(sw), int(sh)
            self._is_fitted = data.get("is_fitted", True)
            logger.info("GazeMapper: migrated legacy pkl from %s", pkl_path)
            return True
        except Exception:
            logger.exception("GazeMapper: error migrating legacy pkl %s", pkl_path)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_features(
        self,
        gaze_angles: np.ndarray,
        head_poses: np.ndarray | None,
    ) -> np.ndarray:
        """Build polynomial feature matrix from gaze angles (+ optional head pose)."""
        yaw = gaze_angles[:, 0:1]
        pitch = gaze_angles[:, 1:2]
        feats: list[np.ndarray] = [yaw, pitch, yaw**2, pitch**2, yaw * pitch]
        if head_poses is not None:
            feats.append(head_poses)
        return np.hstack(feats)

    def _loo_error(
        self,
        X: np.ndarray,
        y_x: np.ndarray,
        y_y: np.ndarray,
    ) -> float:
        """Leave-one-out cross-validation error (pixels)."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        errors: list[float] = []
        n = len(X)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            sc = StandardScaler().fit(X[mask])
            Xs_tr = sc.transform(X[mask])
            rx = Ridge(alpha=1.0).fit(Xs_tr, y_x[mask])
            ry = Ridge(alpha=1.0).fit(Xs_tr, y_y[mask])
            xi = sc.transform(X[i : i + 1])
            errors.append(
                float(np.hypot(rx.predict(xi)[0] - y_x[i], ry.predict(xi)[0] - y_y[i]))
            )
        return float(np.mean(errors))
