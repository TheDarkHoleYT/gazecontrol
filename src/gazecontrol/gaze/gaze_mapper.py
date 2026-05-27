"""GazeMapper — convert L2CS-Net (yaw, pitch) angles to screen pixel coordinates.

Persistence format: ``.npz`` + ``.meta.json`` (version-stable; no sklearn pickle).
Backward-compatible migration from old ``.pkl`` files is provided by
:func:`load_legacy_pkl`.

Schema versions
---------------
- ``"1"`` (v0.7–v0.8): coefficients, intercepts, scaler params, screen size,
  ``is_fitted`` flag. Used by all profiles created before v1.0.
- ``"2"`` (v1.0+, ADR-0009): adds inline training data
  (``training_angles``, ``training_targets``, optional
  ``training_head_poses``) so :meth:`partial_fit` can incrementally
  refit, plus metadata (``calibrated_at``, ``samples_count``,
  ``loo_error_px``, ``holdout_error_px``, ``monitor_id``, ``user_id``,
  ``fit_method``, ``mapper_type``, ``feature_schema``) that supports
  per-user / per-monitor profile management and stale-calibration
  detection. v1 files load unchanged: missing fields default and an
  INFO log line suggests recalibration.

Predict contract:
- Returns ``(px_x, px_y)`` when fitted.
- Returns ``None`` when unfitted — callers must handle this explicitly.
  (Previously returned a noisy geometric estimate without warning.)
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_FORMAT_VERSION = "2"

#: Mapper types known to v1.0. Future kernel/GP variants register here.
_KNOWN_MAPPER_TYPES = frozenset({"poly_ridge", "kernel_ridge", "gp"})

#: Default feature schema for the v1 polynomial-ridge mapper. Stored in
#: meta.json so a future kernel/GP load can reconstruct the same feature
#: vector at predict time.
_POLY_RIDGE_FEATURE_SCHEMA: tuple[str, ...] = (
    "yaw",
    "pitch",
    "yaw_sq",
    "pitch_sq",
    "yaw_pitch",
)
_POLY_RIDGE_HEAD_POSE_FEATURES: tuple[str, ...] = (
    "head_yaw",
    "head_pitch",
    "head_roll",
)


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
        self._coef_x: np.ndarray[Any, Any] | None = None
        self._intercept_x: float = 0.0
        self._coef_y: np.ndarray[Any, Any] | None = None
        self._intercept_y: float = 0.0
        # Scaler parameters (saved as npz arrays).
        self._scaler_mean: np.ndarray[Any, Any] | None = None
        self._scaler_scale: np.ndarray[Any, Any] | None = None
        self._is_fitted: bool = False
        # --- Schema v2 (ADR-0009) -----------------------------------------
        # Training data persisted inline so partial_fit / incremental
        # recalibration can refit without re-running the full grid.
        self._training_angles: np.ndarray[Any, Any] | None = None
        self._training_targets: np.ndarray[Any, Any] | None = None
        self._training_head_poses: np.ndarray[Any, Any] | None = None
        # Profile metadata (mirrored into meta.json).
        self._mapper_type: str = "poly_ridge"
        self._calibrated_at: str | None = None
        self._samples_count: int = 0
        self._loo_error_px: float | None = None
        self._holdout_error_px: float | None = None
        self._monitor_id: str | None = None
        self._user_id: str = "default"
        self._fit_method: str | None = None
        self._feature_schema: list[str] = list(_POLY_RIDGE_FEATURE_SCHEMA)
        # True when the profile was loaded from a pre-v1.0 schema and the
        # missing fields were defaulted. Callers can surface this to the
        # user (e.g. "recalibration recommended").
        self._loaded_from_legacy_v1: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True when the mapper has been trained and can predict."""
        return self._is_fitted

    @property
    def loaded_from_legacy_v1(self) -> bool:
        """True when ``load()`` populated v2 metadata from a v1 profile.

        Callers (HUD, calibration runner) use this to suggest a
        recalibration to the user without forcing it.
        """
        return self._loaded_from_legacy_v1

    def metadata(self) -> dict[str, Any]:
        """Return a copy of the v2 profile metadata as a plain dict.

        Useful for diagnostics, HUD ("calibrated 3 days ago"), and the
        ``--doctor --functional`` command. Returns defaults for every
        field even when the mapper was loaded from a legacy v1 profile.
        """
        return {
            "schema_version": _FORMAT_VERSION,
            "mapper_type": self._mapper_type,
            "calibrated_at": self._calibrated_at,
            "samples_count": self._samples_count,
            "loo_error_px": self._loo_error_px,
            "holdout_error_px": self._holdout_error_px,
            "monitor_id": self._monitor_id,
            "user_id": self._user_id,
            "fit_method": self._fit_method,
            "feature_schema": list(self._feature_schema),
            "screen_w": self._sw,
            "screen_h": self._sh,
            "is_fitted": self._is_fitted,
            "loaded_from_legacy_v1": self._loaded_from_legacy_v1,
        }

    def set_profile_identity(
        self,
        *,
        user_id: str | None = None,
        monitor_id: str | None = None,
    ) -> None:
        """Tag the loaded/fitted mapper with a user/monitor identity.

        Per ADR-0009 the canonical on-disk layout is
        ``<profiles>/<user>/<monitor>/v{N}.npz``; the calibration runner
        and the migrator call this just before :meth:`save` so the
        metadata mirrors the path.
        """
        if user_id is not None:
            self._user_id = user_id
        if monitor_id is not None:
            self._monitor_id = monitor_id

    def fit(
        self,
        gaze_angles: np.ndarray[Any, Any],
        screen_points: np.ndarray[Any, Any],
        head_poses: np.ndarray[Any, Any] | None = None,
        *,
        fit_method: str = "9pt",
        holdout_error_px: float | None = None,
    ) -> float:
        """Fit the mapper on calibration data.

        Args:
            gaze_angles:      (N, 2) array of (yaw, pitch) in degrees.
            screen_points:    (N, 2) array of (px_x, px_y) ground-truth screen coords.
            head_poses:       (N, 3) optional head pose (yaw, pitch, roll) in radians.
            fit_method:       Label for the calibration routine that produced
                              ``gaze_angles`` / ``screen_points`` (e.g. ``"9pt"``,
                              ``"13pt_holdout"``, ``"incremental_3pt"``). Persisted
                              into the v2 schema for audit / UX hints.
            holdout_error_px: Optional held-out validation error in pixels (computed
                              by the calibration runner when a holdout split is
                              available). Persisted alongside the LOO error.

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
        logger.info("GazeMapper fitted: LOO error = %.1f px (%.2f°)", loo_error, loo_error / 44.0)

        # --- Schema v2: persist training data + metadata --------------------
        self._training_angles = np.asarray(gaze_angles, dtype=np.float64).copy()
        self._training_targets = np.asarray(screen_points, dtype=np.float64).copy()
        if head_poses is not None:
            self._training_head_poses = np.asarray(head_poses, dtype=np.float64).copy()
            self._feature_schema = list(_POLY_RIDGE_FEATURE_SCHEMA) + list(
                _POLY_RIDGE_HEAD_POSE_FEATURES
            )
        else:
            self._training_head_poses = None
            self._feature_schema = list(_POLY_RIDGE_FEATURE_SCHEMA)
        self._samples_count = len(gaze_angles)
        self._loo_error_px = float(loo_error)
        self._holdout_error_px = float(holdout_error_px) if holdout_error_px is not None else None
        self._fit_method = fit_method
        self._mapper_type = "poly_ridge"
        self._calibrated_at = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        # _user_id / _monitor_id are set explicitly by the calibration runner
        # before save(); leave whatever the caller configured.
        # Once we've gone through a successful fit, the profile is no longer
        # "legacy" — drop the flag so callers stop nagging the user.
        self._loaded_from_legacy_v1 = False
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

        px_x = float((Xs @ self._coef_x).item() + self._intercept_x)
        px_y = float((Xs @ self._coef_y).item() + self._intercept_y)

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

        # Atomic write: stage both the .npz and the .meta.json under .part
        # filenames, then os.replace() each into place only after BOTH are
        # written.  Avoids leaving the user with a corrupt profile (or only
        # one of the two files) if the process crashes mid-save.
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        meta_path = path.parent / (path.stem + ".meta.json")
        npz_part = npz_path.with_suffix(npz_path.suffix + ".part")
        meta_part = meta_path.with_suffix(meta_path.suffix + ".part")

        try:
            np.savez_compressed(
                str(npz_part),
                coef_x=self._coef_x if self._coef_x is not None else np.array([]),
                coef_y=self._coef_y if self._coef_y is not None else np.array([]),
                intercept_x=np.array([self._intercept_x]),
                intercept_y=np.array([self._intercept_y]),
                scaler_mean=self._scaler_mean if self._scaler_mean is not None else np.array([]),
                scaler_scale=self._scaler_scale if self._scaler_scale is not None else np.array([]),
                # --- Schema v2 arrays (ADR-0009) -----------------------------
                # Empty placeholder arrays are written when the data is not
                # present so np.load() does not raise KeyError on legacy
                # codepaths that look these keys up directly.
                training_angles=(
                    self._training_angles if self._training_angles is not None else np.array([])
                ),
                training_targets=(
                    self._training_targets if self._training_targets is not None else np.array([])
                ),
                training_head_poses=(
                    self._training_head_poses
                    if self._training_head_poses is not None
                    else np.array([])
                ),
            )
            # numpy may add an extra suffix when the path already has one.
            # Resolve the actual file numpy wrote so os.replace() targets the
            # correct stem.
            actual_npz_part = (
                npz_part if npz_part.exists() else npz_part.with_suffix(npz_part.suffix + ".npz")
            )

            meta = {
                "format_version": _FORMAT_VERSION,
                "schema_version": _FORMAT_VERSION,  # alias used by ADR-0009
                "screen_w": self._sw,
                "screen_h": self._sh,
                "is_fitted": self._is_fitted,
                # Schema v2 metadata (ADR-0009).
                "mapper_type": self._mapper_type,
                "calibrated_at": self._calibrated_at,
                "samples_count": self._samples_count,
                "loo_error_px": self._loo_error_px,
                "holdout_error_px": self._holdout_error_px,
                "monitor_id": self._monitor_id,
                "user_id": self._user_id,
                "fit_method": self._fit_method,
                "feature_schema": list(self._feature_schema),
            }
            meta_part.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            os.replace(actual_npz_part, npz_path)
            os.replace(meta_part, meta_path)
        except Exception:
            for stale in (npz_part, npz_part.with_suffix(npz_part.suffix + ".npz"), meta_part):
                if stale.exists():
                    with contextlib.suppress(OSError):
                        stale.unlink()
            raise
        logger.info("GazeMapper saved to %s", npz_path)

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
        meta: dict[str, Any] = {}
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

            # --- Schema-version dispatch (ADR-0009) -------------------------
            # ``schema_version`` is the canonical key from v1.0; older
            # profiles wrote ``format_version`` instead. Treat either as
            # authoritative; "1" or missing → legacy v1 layout.
            schema_raw = meta.get("schema_version") or meta.get("format_version") or "1"
            schema_version = str(schema_raw)

            if schema_version == "1":
                # v1 profile loaded under v1.0+ runtime: populate v2 metadata
                # with conservative defaults and warn the user (once) that a
                # recalibration is recommended to take advantage of the new
                # mapper features. Existing predict() works unchanged.
                self._loaded_from_legacy_v1 = True
                self._mapper_type = "poly_ridge"
                self._calibrated_at = None
                self._samples_count = 0
                self._loo_error_px = None
                self._holdout_error_px = None
                self._monitor_id = None
                self._user_id = meta.get("user_id", "default")
                self._fit_method = "legacy_v1"
                self._feature_schema = list(_POLY_RIDGE_FEATURE_SCHEMA)
                self._training_angles = None
                self._training_targets = None
                self._training_head_poses = None
                logger.info(
                    "GazeMapper: loaded legacy v1 profile from %s; "
                    "recalibration recommended to populate v2 metadata "
                    "(holdout error, training data for partial_fit).",
                    npz_path,
                )
            elif schema_version == "2":
                self._loaded_from_legacy_v1 = False
                self._mapper_type = str(meta.get("mapper_type", "poly_ridge"))
                if self._mapper_type not in _KNOWN_MAPPER_TYPES:
                    logger.warning(
                        "GazeMapper: unknown mapper_type %r in %s; "
                        "treating as poly_ridge.",
                        self._mapper_type,
                        npz_path,
                    )
                    self._mapper_type = "poly_ridge"
                self._calibrated_at = meta.get("calibrated_at")
                self._samples_count = int(meta.get("samples_count", 0) or 0)
                loo = meta.get("loo_error_px")
                self._loo_error_px = float(loo) if loo is not None else None
                hold = meta.get("holdout_error_px")
                self._holdout_error_px = float(hold) if hold is not None else None
                self._monitor_id = meta.get("monitor_id")
                self._user_id = str(meta.get("user_id", "default"))
                self._fit_method = meta.get("fit_method")
                feat = meta.get("feature_schema")
                self._feature_schema = (
                    [str(f) for f in feat]
                    if isinstance(feat, list)
                    else list(_POLY_RIDGE_FEATURE_SCHEMA)
                )
                # Training data (optional — empty placeholders OK).
                ta = data.get("training_angles") if hasattr(data, "get") else None
                if ta is None and "training_angles" in data.files:
                    ta = data["training_angles"]
                self._training_angles = ta if ta is not None and ta.size > 0 else None
                tt = None
                if "training_targets" in data.files:
                    tt = data["training_targets"]
                self._training_targets = tt if tt is not None and tt.size > 0 else None
                th = None
                if "training_head_poses" in data.files:
                    th = data["training_head_poses"]
                self._training_head_poses = th if th is not None and th.size > 0 else None
            else:
                # Future schema (v3+) — refuse to load rather than guess.
                logger.error(
                    "GazeMapper: unsupported schema_version=%r in %s; "
                    "upgrade gazecontrol to read this profile.",
                    schema_version,
                    npz_path,
                )
                return False

            logger.info(
                "GazeMapper loaded from %s (fitted=%s, schema=%s, mapper_type=%s)",
                npz_path,
                self._is_fitted,
                schema_version,
                self._mapper_type,
            )
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
        import pickle

        pkl_path = Path(pkl_path)
        try:
            with pkl_path.open("rb") as fh:
                data = pickle.load(fh)  # noqa: S301  # nosec B301
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
        gaze_angles: np.ndarray[Any, Any],
        head_poses: np.ndarray[Any, Any] | None,
    ) -> np.ndarray[Any, Any]:
        """Build polynomial feature matrix from gaze angles (+ optional head pose)."""
        yaw = gaze_angles[:, 0:1]
        pitch = gaze_angles[:, 1:2]
        feats: list[np.ndarray[Any, Any]] = [yaw, pitch, yaw**2, pitch**2, yaw * pitch]
        if head_poses is not None:
            feats.append(head_poses)
        return np.hstack(feats)

    def _loo_error(
        self,
        X: np.ndarray[Any, Any],  # noqa: N803
        y_x: np.ndarray[Any, Any],
        y_y: np.ndarray[Any, Any],
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
            errors.append(float(np.hypot(rx.predict(xi)[0] - y_x[i], ry.predict(xi)[0] - y_y[i])))
        return float(np.mean(errors))
