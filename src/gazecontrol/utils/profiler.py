"""Pipeline Latency Profiler — misura il tempo di ogni stage del loop principale.

Uso:
    profiler = PipelineProfiler(log_every_n=300)  # logga ogni 10s a 30fps

    with profiler.stage('landmarks'):
        features, blink = estimator.extract_features(frame)

    with profiler.stage('gesture'):
        feat = extractor.extract(hand_result)

    profiler.tick()  # chiama a fine loop
"""
from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class PipelineProfiler:
    """Misura la latenza (ms) di ogni named stage con rolling window.

    Attributes:
        log_every_n : log periodico ogni N frame.
        window_size : campioni tenuti per calcolare la media rolling.
    """

    def __init__(self, log_every_n: int = 300, window_size: int = 60) -> None:
        self._log_every = log_every_n
        self._tick_count = 0
        self._stages: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._stage_start: dict[str, float] = {}

    @contextlib.contextmanager
    def stage(self, name: str):
        """Context manager per misurare un singolo stage."""
        t0 = time.monotonic()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self._stages[name].append(elapsed_ms)

    def tick(self) -> None:
        """Chiama una volta per frame. Logga le statistiche ogni log_every_n frame."""
        self._tick_count += 1
        if self._tick_count % self._log_every == 0:
            self._log_stats()

    def _log_stats(self) -> None:
        if not self._stages:
            return
        parts = []
        total = 0.0
        for name, buf in self._stages.items():
            if buf:
                mean_ms = sum(buf) / len(buf)
                total += mean_ms
                parts.append(f"{name}={mean_ms:.1f}ms")
        parts.append(f"total={total:.1f}ms")
        logger.info("[PERF] %s", " | ".join(parts))

    def stats(self) -> dict[str, float]:
        """Ritorna il dizionario nome→media_ms per uso programmatico."""
        return {
            name: (sum(buf) / len(buf) if buf else 0.0)
            for name, buf in self._stages.items()
        }
