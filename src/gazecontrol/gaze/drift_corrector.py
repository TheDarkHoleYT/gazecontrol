"""DriftCorrector — correzione automatica del drift di calibrazione.

Tre strategie complementari applicate in sequenza:

1. Edge Snapping: quando il gaze supera i bordi dello schermo di oltre margin_px,
   corregge gradualmente l'offset (rileva drift globale sinistra/destra/alto/basso).

2. Implicit Recalibration: quando l'utente esegue un'azione (DRAG/CLOSE/etc.)
   su una finestra, il centroide della finestra target viene usato come "ground truth"
   per stimare il drift attuale → micro-correzione EMA (o gain Kalman v1.0).

3. Explicit Recenter (v1.0, G7): l'utente preme ctrl+shift+r e fissa un punto
   noto (di solito il centro schermo) — la differenza media tra gaze grezzo e
   target diventa l'offset direttamente, senza EMA. Vedi
   :meth:`request_recenter` / :meth:`feed_recenter_sample` /
   :meth:`recenter_to`.

4. Convergence telemetry (v1.0, G7): :meth:`is_converged` traccia la stddev
   delle ultime N variazioni di offset; quando scende sotto la soglia il drift
   è considerato stabile e l'HUD può comunicarlo all'utente.
"""

from __future__ import annotations

import collections
import logging
import math
from typing import Any, Literal

logger = logging.getLogger(__name__)

#: Active update mode for implicit recalibration observations.
DriftMode = Literal["ema", "kalman"]


class DriftCorrector:
    """Corregge il drift del gaze point tramite strategie implicite o esplicite.

    Args:
        screen_w, screen_h:    Dimensioni schermo in pixel.
        edge_margin_px:        Margine oltre cui scatta l'edge snapping.
        edge_correction_rate:  EMA alpha per edge snapping (0 < r < 1).
        implicit_alpha:        EMA alpha per implicit recalibration.
        max_correction_px:     Cap massimo correzione totale in px.
        mode:                  ``"ema"`` (legacy) oppure ``"kalman"`` (v1.0).
        recenter_sample_count: Quanti sample raccoglie ``request_recenter``
                               prima di chiudere il flusso e fissare l'offset.
        convergence_window:    Quanti delta di offset alimentano
                               :meth:`is_converged`.
        convergence_threshold_px: Stddev sotto cui l'offset è "convergente".
        kalman_process_noise:  Q (px²) — varianza di processo dello stato
                               offset. Più grande = inseguimento più rapido.
        kalman_measurement_noise: R (px²) — varianza per ogni singola
                               osservazione di recal.
    """

    def __init__(
        self,
        screen_w: int = 1920,
        screen_h: int = 1080,
        edge_margin_px: int = 60,
        edge_correction_rate: float = 0.05,
        implicit_alpha: float = 0.08,
        max_correction_px: float = 120.0,
        *,
        mode: DriftMode = "ema",
        recenter_sample_count: int = 30,
        convergence_window: int = 60,
        convergence_threshold_px: float = 5.0,
        kalman_process_noise: float = 0.5,
        kalman_measurement_noise: float = 50.0,
    ) -> None:
        self._sw = screen_w
        self._sh = screen_h
        self._margin = edge_margin_px
        self._edge_rate = edge_correction_rate
        self._impl_alpha = implicit_alpha
        self._max_corr = max_correction_px
        self._mode: DriftMode = mode

        # Offset corrente (px): viene sottratto al raw gaze point.
        self._offset_x = 0.0
        self._offset_y = 0.0

        # Kalman state: per-axis covariance P (px²). Initialised optimistic
        # so the first observation pulls the offset strongly.
        self._cov_x = float(kalman_measurement_noise)
        self._cov_y = float(kalman_measurement_noise)
        self._kalman_q = float(kalman_process_noise)
        self._kalman_r = float(kalman_measurement_noise)

        # Convergence telemetry: rolling window of offset deltas.
        self._conv_window = int(convergence_window)
        self._conv_threshold = float(convergence_threshold_px)
        self._delta_history: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=self._conv_window
        )

        # Explicit-recenter pipeline state (G7).
        self._recenter_sample_count = int(recenter_sample_count)
        self._recenter_target: tuple[float, float] | None = None
        self._recenter_samples: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def correct(self, x: float, y: float) -> tuple[float, float]:
        """Applica la correzione drift al punto gaze grezzo.

        Returns:
            (x_corrected, y_corrected) clampato ai bordi schermo.
        """
        cx = x - self._offset_x
        cy = y - self._offset_y

        # Edge snapping: se il punto corretto è ancora fuori, aggiusta offset.
        self._update_edge_snapping(x, y)

        cx = max(0.0, min(float(self._sw - 1), cx))
        cy = max(0.0, min(float(self._sh - 1), cy))
        return cx, cy

    def on_action(
        self,
        gaze_point: tuple[float, float],
        target_window: dict[str, Any],
    ) -> None:
        """Update drift estimate after a user action on a window.

        Uses the window centroid as gaze ground-truth.

        Args:
            gaze_point: Gaze point at action time (after ``correct()``).
            target_window: Window info dict with ``'rect'`` = (x, y, w, h).
        """
        rect = target_window.get("rect")
        if not rect:
            return
        win_cx = rect[0] + rect[2] / 2.0
        win_cy = rect[1] + rect[3] / 2.0

        # Stima errore: il gaze dovrebbe puntare al centroide della finestra.
        err_x = gaze_point[0] - win_cx
        err_y = gaze_point[1] - win_cy
        self._apply_observation(err_x, err_y)

        logger.debug(
            "DriftCorrector: implicit recal err=(%.1f,%.1f) offset=(%.1f,%.1f) mode=%s",
            err_x,
            err_y,
            self._offset_x,
            self._offset_y,
            self._mode,
        )

    def reset(self) -> None:
        """Azzera la correzione (utile dopo una nuova calibrazione)."""
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._cov_x = self._kalman_r
        self._cov_y = self._kalman_r
        self._delta_history.clear()
        self._recenter_target = None
        self._recenter_samples.clear()
        logger.debug("DriftCorrector: offset azzerato")

    @property
    def offset(self) -> tuple[float, float]:
        """Current drift offset as (dx, dy) in pixels."""
        return (self._offset_x, self._offset_y)

    @property
    def mode(self) -> DriftMode:
        """Active update mode (``"ema"`` or ``"kalman"``)."""
        return self._mode

    # ------------------------------------------------------------------
    # Convergence telemetry (G7)
    # ------------------------------------------------------------------

    def is_converged(self) -> bool:
        """True when the offset state has stabilised.

        Computed as ``stddev(last N offset deltas) < threshold_px`` on both
        axes. Always False until the window is full so the HUD does not
        flash "converged" during the warm-up phase.
        """
        if len(self._delta_history) < self._conv_window:
            return False
        dx = [d[0] for d in self._delta_history]
        dy = [d[1] for d in self._delta_history]
        return _stddev(dx) < self._conv_threshold and _stddev(dy) < self._conv_threshold

    def telemetry(self) -> dict[str, float | int | bool | str]:
        """Snapshot of drift state for HUD / structured-log telemetry.

        Returns a dict with the offset magnitude, convergence flag, the
        active mode, and the Kalman covariance trace. Cheap to call once
        per frame.
        """
        return {
            "offset_x_px": self._offset_x,
            "offset_y_px": self._offset_y,
            "offset_mag_px": math.hypot(self._offset_x, self._offset_y),
            "mode": self._mode,
            "samples_seen": len(self._delta_history),
            "converged": self.is_converged(),
            "cov_x": self._cov_x,
            "cov_y": self._cov_y,
        }

    # ------------------------------------------------------------------
    # Explicit recenter (G7)
    # ------------------------------------------------------------------

    def request_recenter(self, target_xy: tuple[float, float] | None = None) -> None:
        """Open an explicit-recenter window.

        Subsequent calls to :meth:`feed_recenter_sample` collect raw-gaze
        samples until ``recenter_sample_count`` is reached, at which
        point the mean ``raw − target`` becomes the new offset directly
        (no EMA). If *target_xy* is ``None`` the screen centre is used.

        Calling this while another recenter is in flight resets the
        collection — the user clearly changed their mind.
        """
        if target_xy is None:
            target_xy = (self._sw / 2.0, self._sh / 2.0)
        self._recenter_target = target_xy
        self._recenter_samples = []
        logger.info(
            "DriftCorrector: recenter requested at target=(%.0f,%.0f), need %d samples",
            target_xy[0],
            target_xy[1],
            self._recenter_sample_count,
        )

    def feed_recenter_sample(self, raw_xy: tuple[float, float]) -> bool:
        """Feed a raw gaze sample into an active recenter session.

        Returns:
            True when the session is now complete (offset updated);
            False when more samples are needed; False also when no
            recenter is currently active.
        """
        if self._recenter_target is None:
            return False
        self._recenter_samples.append(raw_xy)
        if len(self._recenter_samples) < self._recenter_sample_count:
            return False
        # Enough samples — finalize.
        target = self._recenter_target
        avg_x = sum(s[0] for s in self._recenter_samples) / len(self._recenter_samples)
        avg_y = sum(s[1] for s in self._recenter_samples) / len(self._recenter_samples)
        self.recenter_to((avg_x, avg_y), target)
        self._recenter_target = None
        self._recenter_samples = []
        return True

    def recenter_to(
        self,
        raw_xy: tuple[float, float],
        target_xy: tuple[float, float],
    ) -> None:
        """One-shot direct offset assignment.

        Sets ``offset = raw_xy − target_xy`` directly, then clamps to
        ``max_correction_px``. Resets the Kalman covariance to ``R`` so
        future implicit-recal observations weigh against fresh
        confidence rather than years of accumulated certainty.
        """
        new_offset_x = raw_xy[0] - target_xy[0]
        new_offset_y = raw_xy[1] - target_xy[1]
        prev = (self._offset_x, self._offset_y)
        self._offset_x = new_offset_x
        self._offset_y = new_offset_y
        self._cov_x = self._kalman_r
        self._cov_y = self._kalman_r
        self._clamp_offset()
        self._delta_history.append(
            (self._offset_x - prev[0], self._offset_y - prev[1])
        )
        logger.info(
            "DriftCorrector: recenter complete — offset=(%.1f,%.1f)",
            self._offset_x,
            self._offset_y,
        )

    @property
    def recenter_in_progress(self) -> bool:
        """True between ``request_recenter()`` and the final sample."""
        return self._recenter_target is not None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_observation(self, err_x: float, err_y: float) -> None:
        """Fold a single (err_x, err_y) observation into the offset state."""
        prev_x = self._offset_x
        prev_y = self._offset_y
        if self._mode == "kalman":
            # Predict step: covariance grows by Q.
            self._cov_x += self._kalman_q
            self._cov_y += self._kalman_q
            # Update step: gain K = P / (P + R).
            kx = self._cov_x / (self._cov_x + self._kalman_r)
            ky = self._cov_y / (self._cov_y + self._kalman_r)
            self._offset_x += kx * err_x
            self._offset_y += ky * err_y
            self._cov_x = (1.0 - kx) * self._cov_x
            self._cov_y = (1.0 - ky) * self._cov_y
        else:
            self._offset_x += self._impl_alpha * err_x
            self._offset_y += self._impl_alpha * err_y
        self._clamp_offset()
        self._delta_history.append(
            (self._offset_x - prev_x, self._offset_y - prev_y)
        )

    def _update_edge_snapping(self, raw_x: float, raw_y: float) -> None:
        """Adjust offset toward zero when gaze crosses screen boundaries.

        ``corrected = raw - offset``.  When ``corrected`` overshoots a screen
        edge by more than ``margin`` we infer a systematic bias in that
        direction and shift ``offset`` so the next ``corrected`` lands closer
        to the edge.  Mirrors the sign convention of :meth:`on_action`:

            err = corrected - target_edge
            offset += rate * err
        """
        prev_x = self._offset_x
        prev_y = self._offset_y
        corrected_x = raw_x - self._offset_x
        if corrected_x < -self._margin:
            # corrected too far LEFT → err < 0 → decrease offset_x.
            self._offset_x -= self._edge_rate * abs(corrected_x)
        elif corrected_x > self._sw + self._margin:
            # corrected too far RIGHT → err > 0 → increase offset_x.
            self._offset_x += self._edge_rate * abs(corrected_x - self._sw)

        corrected_y = raw_y - self._offset_y
        if corrected_y < -self._margin:
            self._offset_y -= self._edge_rate * abs(corrected_y)
        elif corrected_y > self._sh + self._margin:
            self._offset_y += self._edge_rate * abs(corrected_y - self._sh)

        self._clamp_offset()
        dx = self._offset_x - prev_x
        dy = self._offset_y - prev_y
        if dx != 0.0 or dy != 0.0:
            self._delta_history.append((dx, dy))

    def _clamp_offset(self) -> None:
        mag = math.hypot(self._offset_x, self._offset_y)
        if mag > self._max_corr and mag > 0:
            scale = self._max_corr / mag
            self._offset_x *= scale
            self._offset_y *= scale


def _stddev(xs: list[float]) -> float:
    """Population stddev of *xs*. Returns 0 for n<2."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return math.sqrt(sum((v - mean) ** 2 for v in xs) / n)
