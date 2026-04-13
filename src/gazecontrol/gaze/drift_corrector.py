"""
DriftCorrector — correzione automatica del drift di calibrazione.

Tre strategie complementari applicate in sequenza:

1. Edge Snapping: quando il gaze supera i bordi dello schermo di oltre margin_px,
   corregge gradualmente l'offset (rileva drift globale sinistra/destra/alto/basso).

2. Implicit Recalibration: quando l'utente esegue un'azione (DRAG/CLOSE/etc.)
   su una finestra, il centroide della finestra target viene usato come "ground truth"
   per stimare il drift attuale → micro-correzione EMA.

3. Screen-edge Prior: durante le fixation ai bordi (taskbar, titlebars), applica
   attrazione verso gli snap points. Disabilitato per default perché richiede
   screen layout awareness.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DriftCorrector:
    """
    Corregge il drift del gaze point tramite strategie implicite.

    Args:
        screen_w, screen_h : dimensioni schermo in pixel.
        edge_margin_px     : margine oltre cui scatta l'edge snapping.
        edge_correction_rate : EMA alpha per edge snapping (0 < r < 1).
        implicit_alpha     : EMA alpha per implicit recalibration.
        max_correction_px  : cap massimo correzione totale in px.
    """

    def __init__(
        self,
        screen_w: int = 1920,
        screen_h: int = 1080,
        edge_margin_px: int = 60,
        edge_correction_rate: float = 0.05,
        implicit_alpha: float = 0.08,
        max_correction_px: float = 120.0,
    ):
        self._sw = screen_w
        self._sh = screen_h
        self._margin = edge_margin_px
        self._edge_rate = edge_correction_rate
        self._impl_alpha = implicit_alpha
        self._max_corr = max_correction_px

        # Offset corrente (px): viene sottratto al raw gaze point
        self._offset_x = 0.0
        self._offset_y = 0.0

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def correct(self, x: float, y: float) -> tuple[float, float]:
        """
        Applica la correzione drift al punto gaze grezzo.

        Returns:
            (x_corrected, y_corrected) clampato ai bordi schermo.
        """
        cx = x - self._offset_x
        cy = y - self._offset_y

        # Edge snapping: se il punto corretto è ancora fuori, aggiusta offset
        self._update_edge_snapping(x, y)

        cx = max(0.0, min(float(self._sw - 1), cx))
        cy = max(0.0, min(float(self._sh - 1), cy))
        return cx, cy

    def on_action(self, gaze_point: tuple[float, float],
                  target_window: dict) -> None:
        """
        Chiamato quando viene eseguita un'azione su una finestra.
        Usa il centroide della finestra come ground truth per il drift.

        Args:
            gaze_point    : punto gaze al momento dell'azione (dopo correct()).
            target_window : dict con chiave 'rect' = (x, y, w, h).
        """
        rect = target_window.get('rect')
        if not rect:
            return
        win_cx = rect[0] + rect[2] / 2.0
        win_cy = rect[1] + rect[3] / 2.0

        # Stima errore: il gaze dovrebbe puntare al centroide della finestra
        err_x = gaze_point[0] - win_cx
        err_y = gaze_point[1] - win_cy

        # Aggiorna offset con EMA (piccola correzione graduale)
        self._offset_x += self._impl_alpha * err_x
        self._offset_y += self._impl_alpha * err_y
        self._clamp_offset()

        logger.debug("DriftCorrector: implicit recal err=(%.1f,%.1f) "
                     "offset=(%.1f,%.1f)", err_x, err_y,
                     self._offset_x, self._offset_y)

    def reset(self):
        """Azzera la correzione (utile dopo una nuova calibrazione)."""
        self._offset_x = 0.0
        self._offset_y = 0.0
        logger.debug("DriftCorrector: offset azzerato")

    @property
    def offset(self) -> tuple[float, float]:
        return (self._offset_x, self._offset_y)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_edge_snapping(self, raw_x: float, raw_y: float) -> None:
        """
        Se il gaze grezzo supera i bordi dello schermo → aggiusta offset verso zero.
        Questo rilevamento indica drift sistematico in quella direzione.
        """
        if raw_x - self._offset_x < -self._margin:
            # Gaze va troppo a sinistra → correzione positiva su x
            self._offset_x += self._edge_rate * abs(raw_x - self._offset_x)
        elif raw_x - self._offset_x > self._sw + self._margin:
            self._offset_x -= self._edge_rate * abs(raw_x - self._offset_x - self._sw)

        if raw_y - self._offset_y < -self._margin:
            self._offset_y += self._edge_rate * abs(raw_y - self._offset_y)
        elif raw_y - self._offset_y > self._sh + self._margin:
            self._offset_y -= self._edge_rate * abs(raw_y - self._offset_y - self._sh)

        self._clamp_offset()

    def _clamp_offset(self):
        import math
        mag = math.hypot(self._offset_x, self._offset_y)
        if mag > self._max_corr and mag > 0:
            scale = self._max_corr / mag
            self._offset_x *= scale
            self._offset_y *= scale
