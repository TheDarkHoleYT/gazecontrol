"""Pipeline Latency Profiler — per-stage timing with Prometheus export.

Usage::

    profiler = PipelineProfiler(log_every_n=300, fps_budget=30)

    with profiler.stage("capture"):
        frame_bgr = grabber.read_bgr()

    with profiler.stage("gesture"):
        ctx = gesture_stage.process(ctx)

    profiler.tick()  # call once per frame; logs stats every log_every_n ticks

Prometheus text format
----------------------
Call ``profiler.emit_prometheus(path)`` to write a Prometheus-compatible
``.prom`` file that can be scraped by a node exporter or manually inspected::

    profiler.emit_prometheus(Paths.logs() / "metrics.prom")

The file contains gauge metrics for each stage's p50, p95, and mean latency::

    gazecontrol_stage_latency_ms{stage="capture",quantile="p50"} 2.3
    gazecontrol_stage_latency_ms{stage="capture",quantile="p95"} 4.1
    gazecontrol_stage_latency_ms{stage="capture",quantile="mean"} 2.5
    gazecontrol_stage_total_ms{quantile="mean"} 28.7
"""

from __future__ import annotations

import contextlib
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical stage names — used for per-stage breakdown logging.
STAGE_CAPTURE = "capture"
STAGE_HAND_DETECT = "hand_detect"
STAGE_FEATURES = "features"
STAGE_FILTER = "filter"
STAGE_FSM = "fsm"
STAGE_ACTION = "action"

# Consecutive over-budget frames before a WARNING is emitted.
_BUDGET_WARN_STREAK = 5


class PipelineProfiler:
    """Per-stage rolling-window latency profiler with Prometheus export.

    Args:
        log_every_n:  Emit periodic stats log every N ticks (~10 s at 30 fps).
        window_size:  Rolling window depth (number of recent samples kept).
        fps_budget:   Target pipeline FPS; frames that exceed the budget
                      (1000 / fps_budget ms total) are counted.  A WARNING
                      is logged after :data:`_BUDGET_WARN_STREAK` consecutive
                      over-budget frames.
    """

    def __init__(
        self,
        log_every_n: int = 300,
        window_size: int = 60,
        fps_budget: int = 30,
    ) -> None:
        self._log_every = log_every_n
        self._window_size = window_size
        self._budget_ms = 1000.0 / max(fps_budget, 1)
        self._tick_count = 0
        self._over_budget_streak = 0

        self._stages: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.Lock()

        # Counters / gauges (incremented atomically while ``_lock`` is held).
        self._frames_total: int = 0
        self._frames_dropped_total: int = 0
        self._stage_errors_total: dict[str, int] = defaultdict(int)
        self._camera_restarts_total: int = 0
        self._model_load_failures_total: int = 0
        self._gauge_actual_fps: float = 0.0
        self._gauge_gaze_confidence: float = 0.0
        self._gauge_hand_confidence: float = 0.0

    # ------------------------------------------------------------------
    # Counter / gauge mutators (called from PipelineEngine + stages)
    # ------------------------------------------------------------------

    def inc_frame(self, *, dropped: bool = False) -> None:
        """Increment the per-frame counter (and the dropped counter if applicable)."""
        with self._lock:
            self._frames_total += 1
            if dropped:
                self._frames_dropped_total += 1

    def inc_stage_error(self, stage: str) -> None:
        """Record a stage process() exception."""
        with self._lock:
            self._stage_errors_total[stage] += 1

    def inc_camera_restart(self) -> None:
        """Record a camera restart attempt."""
        with self._lock:
            self._camera_restarts_total += 1

    def inc_model_load_failure(self) -> None:
        """Record a failed model load."""
        with self._lock:
            self._model_load_failures_total += 1

    def set_actual_fps(self, fps: float) -> None:
        """Update the actual-FPS gauge."""
        with self._lock:
            self._gauge_actual_fps = float(fps)

    def set_confidence(
        self,
        *,
        gaze: float | None = None,
        hand: float | None = None,
    ) -> None:
        """Update gaze / hand confidence gauges (None leaves the gauge unchanged)."""
        with self._lock:
            if gaze is not None:
                self._gauge_gaze_confidence = float(gaze)
            if hand is not None:
                self._gauge_hand_confidence = float(hand)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Context manager — measure the duration of a named stage block."""
        # perf_counter has sub-microsecond resolution on all supported platforms;
        # monotonic is ~15.6 ms on Windows, too coarse for sub-frame stage timing.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._stages[name].append(elapsed_ms)

    def tick(self) -> None:
        """Call once per frame.

        Increments the tick counter, checks the total-frame budget, and logs
        stats periodically.
        """
        with self._lock:
            self._tick_count += 1
            should_log = self._tick_count % self._log_every == 0
            # Compute total for this frame from the most-recent sample of each stage.
            total_ms = sum(buf[-1] for buf in self._stages.values() if buf)

        # Budget check (outside the lock to avoid holding it during log).
        if total_ms > self._budget_ms:
            self._over_budget_streak += 1
            if self._over_budget_streak >= _BUDGET_WARN_STREAK:
                logger.warning(
                    "[PERF] Pipeline over budget: %.1f ms > %.1f ms budget "
                    "(%d consecutive frames).",
                    total_ms,
                    self._budget_ms,
                    self._over_budget_streak,
                )
                self._over_budget_streak = 0  # reset after warning
        else:
            self._over_budget_streak = 0

        if should_log:
            self._log_stats()

    # ------------------------------------------------------------------
    # Stats access
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, float]:
        """Return ``{stage_name: mean_ms}`` for all measured stages."""
        with self._lock:
            return {
                name: (sum(buf) / len(buf) if buf else 0.0) for name, buf in self._stages.items()
            }

    def percentiles(self) -> dict[str, dict[str, float]]:
        """Return ``{stage_name: {p50, p95, mean}}`` for all measured stages."""
        with self._lock:
            snapshot = {name: list(buf) for name, buf in self._stages.items()}

        result: dict[str, dict[str, float]] = {}
        for name, samples in snapshot.items():
            if not samples:
                continue
            sorted_s = sorted(samples)
            n = len(sorted_s)
            p50_idx = max(0, int(n * 0.50) - 1)
            p95_idx = max(0, int(n * 0.95) - 1)
            result[name] = {
                "p50": sorted_s[p50_idx],
                "p95": sorted_s[p95_idx],
                "mean": statistics.mean(sorted_s),
            }
        return result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_stats(self) -> None:
        with self._lock:
            snapshot = {name: list(buf) for name, buf in self._stages.items()}
        if not snapshot:
            return

        parts: list[str] = []
        total = 0.0
        for name, samples in snapshot.items():
            if samples:
                mean_ms = sum(samples) / len(samples)
                total += mean_ms
                parts.append(f"{name}={mean_ms:.1f}ms")
        parts.append(f"total={total:.1f}ms")
        logger.info("[PERF] %s", " | ".join(parts))

    # ------------------------------------------------------------------
    # Prometheus export
    # ------------------------------------------------------------------

    def emit_prometheus(self, path: Path) -> None:
        """Write a Prometheus text-format metrics file to *path*.

        The file is safe to read by node_exporter's textfile collector.
        On failure the error is logged and the call returns silently.

        Metrics emitted::

            gazecontrol_stage_latency_ms{stage="<name>",quantile="p50"} <value>
            gazecontrol_stage_latency_ms{stage="<name>",quantile="p95"} <value>
            gazecontrol_stage_latency_ms{stage="<name>",quantile="mean"} <value>
            gazecontrol_total_latency_ms{quantile="mean"} <value>
            gazecontrol_tick_count <value>

        Args:
            path: Destination file path (parent directory must exist).
        """
        pct = self.percentiles()
        if not pct:
            return

        lines: list[str] = [
            "# HELP gazecontrol_stage_latency_ms Pipeline stage latency in milliseconds.",
            "# TYPE gazecontrol_stage_latency_ms gauge",
        ]
        total_mean = 0.0
        for name, stats in pct.items():
            for q in ("p50", "p95", "mean"):
                lines.append(
                    f'gazecontrol_stage_latency_ms{{stage="{name}",quantile="{q}"}} {stats[q]:.3f}'
                )
            total_mean += stats["mean"]

        with self._lock:
            stage_errors = dict(self._stage_errors_total)
            frames_total = self._frames_total
            frames_dropped = self._frames_dropped_total
            cam_restarts = self._camera_restarts_total
            model_fails = self._model_load_failures_total
            gauge_fps = self._gauge_actual_fps
            gauge_gaze = self._gauge_gaze_confidence
            gauge_hand = self._gauge_hand_confidence

        lines += [
            "# HELP gazecontrol_total_latency_ms Total pipeline latency per frame.",
            "# TYPE gazecontrol_total_latency_ms gauge",
            f'gazecontrol_total_latency_ms{{quantile="mean"}} {total_mean:.3f}',
            "# HELP gazecontrol_tick_count Total pipeline frames processed.",
            "# TYPE gazecontrol_tick_count counter",
            f"gazecontrol_tick_count {self._tick_count}",
            "# HELP gazecontrol_frames_total Frames produced by the pipeline.",
            "# TYPE gazecontrol_frames_total counter",
            f"gazecontrol_frames_total {frames_total}",
            "# HELP gazecontrol_frames_dropped_total Frames lost upstream of the pipeline.",
            "# TYPE gazecontrol_frames_dropped_total counter",
            f"gazecontrol_frames_dropped_total {frames_dropped}",
            "# HELP gazecontrol_camera_restarts_total Camera reopen attempts.",
            "# TYPE gazecontrol_camera_restarts_total counter",
            f"gazecontrol_camera_restarts_total {cam_restarts}",
            "# HELP gazecontrol_model_load_failures_total ML model load failures.",
            "# TYPE gazecontrol_model_load_failures_total counter",
            f"gazecontrol_model_load_failures_total {model_fails}",
            "# HELP gazecontrol_pipeline_actual_fps Most recent measured FPS.",
            "# TYPE gazecontrol_pipeline_actual_fps gauge",
            f"gazecontrol_pipeline_actual_fps {gauge_fps:.2f}",
            "# HELP gazecontrol_gaze_confidence Most recent gaze confidence.",
            "# TYPE gazecontrol_gaze_confidence gauge",
            f"gazecontrol_gaze_confidence {gauge_gaze:.3f}",
            "# HELP gazecontrol_hand_confidence Most recent hand confidence.",
            "# TYPE gazecontrol_hand_confidence gauge",
            f"gazecontrol_hand_confidence {gauge_hand:.3f}",
            "# HELP gazecontrol_stage_errors_total Stage process() exceptions.",
            "# TYPE gazecontrol_stage_errors_total counter",
        ]
        for stage, n in stage_errors.items():
            lines.append(f'gazecontrol_stage_errors_total{{stage="{stage}"}} {n}')

        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            logger.debug(
                "PipelineProfiler: could not write Prometheus file %s.", path, exc_info=True
            )
