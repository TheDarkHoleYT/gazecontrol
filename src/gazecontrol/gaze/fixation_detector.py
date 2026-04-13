"""Fixation & Saccade Detector — algoritmo I-VT (Velocity-Threshold Identification).

Classifica ogni campione gaze in una di tre categorie:
  - FIXATION  : sguardo fermo su un punto (bassa velocità angolare < 30°/s)
  - SACCADE   : movimento rapido degli occhi (velocità > 100°/s)
  - PURSUIT   : smooth pursuit, tra fixation e saccade

Valori di soglia standard per eye tracking clinico:
  - Fixation < 30°/s
  - Saccade  > 100°/s

A 60cm dalla webcam e schermo FHD da 24":
  1 pixel ≈ 0.025° → 30°/s ≈ 1200 px/s; 100°/s ≈ 4000 px/s.
  Adattiamo dinamicamente in base alla risoluzione.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class GazeEvent:
    """Classified gaze event from the I-VT fixation detector."""

    type: str           # 'fixation' | 'saccade' | 'pursuit' | 'blink'
    point: tuple[float, float]
    velocity_px_s: float
    duration_s: float = 0.0
    centroid: tuple[float, float] | None = None  # solo per fixation con history


class FixationDetector:
    """Classifica ogni campione gaze in fixation/saccade/pursuit via I-VT.

    Args:
        screen_px_per_degree : pixel per grado visivo (default 44 ≈ 24" FHD a 60cm).
        fixation_vel_thr     : soglia velocità fixation (gradi/s).
        saccade_vel_thr      : soglia velocità saccade (gradi/s).
        fixation_history_ms  : finestra temporale per calcolo centroide fixation (ms).
    """

    FIXATION = 'fixation'
    SACCADE = 'saccade'
    PURSUIT = 'pursuit'
    BLINK = 'blink'

    def __init__(
        self,
        screen_px_per_degree: float = 44.0,
        fixation_vel_thr: float = 30.0,
        saccade_vel_thr: float = 100.0,
        fixation_history_ms: float = 150.0,
    ) -> None:
        self._px_per_deg = screen_px_per_degree
        self._fix_thr = fixation_vel_thr * screen_px_per_degree   # → px/s
        self._sac_thr = saccade_vel_thr * screen_px_per_degree    # → px/s
        self._history_s = fixation_history_ms / 1000.0

        self._prev_point: tuple[float, float] | None = None
        self._prev_ts: float | None = None

        # Buffer per centroide fixation
        self._fix_buffer: deque[tuple[float, float, float]] = deque()  # (x, y, ts)

        self._current_type = self.FIXATION
        self._current_start = time.monotonic()

    def update(self, x: float, y: float, timestamp: float | None = None,
               is_blink: bool = False) -> GazeEvent:
        """Update the detector with a new gaze sample.

        Args:
            x: Gaze x-coordinate in screen pixels.
            y: Gaze y-coordinate in screen pixels.
            timestamp: Time in seconds (default: ``time.monotonic()``).
            is_blink: True when the frame is classified as a blink.

        Returns:
            GazeEvent with type, point, velocity and duration of the current event.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        if is_blink:
            self._prev_point = None
            self._prev_ts = None
            self._fix_buffer.clear()
            dur = timestamp - self._current_start
            self._current_type = self.BLINK
            self._current_start = timestamp
            return GazeEvent(type=self.BLINK, point=(x, y), velocity_px_s=0.0,
                             duration_s=dur)

        velocity = 0.0
        if self._prev_point is not None and self._prev_ts is not None:
            dt = timestamp - self._prev_ts
            if dt > 0.0:
                dx = x - self._prev_point[0]
                dy = y - self._prev_point[1]
                velocity = ((dx**2 + dy**2) ** 0.5) / dt

        # Classifica
        if velocity < self._fix_thr:
            event_type = self.FIXATION
        elif velocity > self._sac_thr:
            event_type = self.SACCADE
        else:
            event_type = self.PURSUIT

        # Aggiorna history fixation
        self._fix_buffer.append((x, y, timestamp))
        # Rimuovi campioni più vecchi della finestra
        cutoff = timestamp - self._history_s
        while self._fix_buffer and self._fix_buffer[0][2] < cutoff:
            self._fix_buffer.popleft()

        # Durante saccade svuota il buffer (non ha senso accumulare)
        if event_type == self.SACCADE:
            self._fix_buffer.clear()

        # Calcola centroide se siamo in fixation
        centroid = None
        if event_type == self.FIXATION and len(self._fix_buffer) >= 3:
            xs = [p[0] for p in self._fix_buffer]
            ys = [p[1] for p in self._fix_buffer]
            centroid = (sum(xs) / len(xs), sum(ys) / len(ys))

        # Durata evento corrente
        if event_type != self._current_type:
            self._current_start = timestamp
            self._current_type = event_type
        duration = timestamp - self._current_start

        self._prev_point = (x, y)
        self._prev_ts = timestamp

        return GazeEvent(
            type=event_type,
            point=(x, y),
            velocity_px_s=velocity,
            duration_s=duration,
            centroid=centroid,
        )

    def reset(self) -> None:
        """Reset detector state (e.g. after calibration or profile change)."""
        self._prev_point = None
        self._prev_ts = None
        self._fix_buffer.clear()
        self._current_type = self.FIXATION
        self._current_start = time.monotonic()
