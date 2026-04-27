"""PipelineEngine — synchronous, Qt-free pipeline runner.

The engine owns the ordered list of stages, the frame loop, and the profiler.
It is designed to be embedded in both a QThread adapter (``QtPipelineThread``)
and a blocking headless runner (``run_blocking``), ensuring a single
implementation path for both modes.

Threading model:
    PipelineEngine.run() must be called from a single dedicated thread.
    It is NOT thread-safe to call run() from multiple threads simultaneously.
    ``stop()`` IS safe to call from any thread (uses threading.Event).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Literal

from gazecontrol.errors import PipelineStageError
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.stage import PipelineStage
from gazecontrol.settings import AppSettings
from gazecontrol.utils.profiler import PipelineProfiler

logger = logging.getLogger(__name__)


class PipelineEngine:
    """Runs an ordered sequence of ``PipelineStage`` instances at a target FPS.

    Emits processed frames via an optional ``on_frame`` callback (called in
    the engine thread — callers must marshal to their own threads if needed).

    Args:
        stages:       Ordered list of pipeline stages to execute each tick.
        settings:     Application settings snapshot (frozen at construction).
        on_frame:     Optional callback ``(ctx: FrameContext) -> None`` called
                      after all stages for each successfully captured tick.
        on_shutdown:  Optional callback called when the loop exits cleanly.
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        settings: AppSettings,
        on_frame: Callable[[FrameContext], None] | None = None,
        on_shutdown: Callable[[], None] | None = None,
        error_policy: Literal["continue", "halt"] = "continue",
    ) -> None:
        self._stages = stages
        self._settings = settings
        self._on_frame = on_frame
        self._on_shutdown = on_shutdown
        self._error_policy: Literal["continue", "halt"] = error_policy
        self._fps = settings.camera.fps
        self._profiler = PipelineProfiler(log_every_n=settings.logging.profiler_log_every_n)

        self._stop_event = threading.Event()

        self._fps_counter = 0
        self._fps_timer = time.monotonic()
        self._actual_fps = 0.0

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Signal the engine to exit at the end of the current tick.

        Thread-safe.
        """
        self._stop_event.set()

    def set_on_frame(self, cb: Callable[[FrameContext], None] | None) -> None:
        """Set (or replace) the per-frame callback.

        Prefer passing the callback via the constructor when possible.
        """
        self._on_frame = cb

    def set_on_shutdown(self, cb: Callable[[], None] | None) -> None:
        """Set (or replace) the shutdown callback.

        Prefer passing the callback via the constructor when possible.
        """
        self._on_shutdown = cb

    @property
    def is_running(self) -> bool:
        """True while the loop has not been asked to stop."""
        return not self._stop_event.is_set()

    @property
    def actual_fps(self) -> float:
        """Last measured frames-per-second."""
        return self._actual_fps

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def start_stages(self) -> bool:
        """Call ``start()`` on every stage in order.

        Returns True when all stages started successfully.  On failure,
        already-started stages are stopped via ``stop_stages()``.
        """
        started: list[PipelineStage] = []
        for stage in self._stages:
            try:
                ok = stage.start()
                if not ok:
                    logger.error("Stage '%s' failed to start.", stage.name)
                    self.stop_stages(started)
                    return False
                started.append(stage)
            except Exception:
                logger.exception("Stage '%s' raised during start().", stage.name)
                self.stop_stages(started)
                return False
        return True

    def stop_stages(self, stages: list[PipelineStage] | None = None) -> None:
        """Call ``stop()`` on stages in reverse order, tolerating failures.

        When *stages* is None, stops all registered stages in reverse order.
        When *stages* is provided (e.g. the subset already started), stops
        only those in reverse order.
        """
        target = stages if stages is not None else self._stages
        for stage in reversed(target):
            self._safe_stop(stage)

    def _safe_stop(self, stage: PipelineStage) -> None:
        try:
            stage.stop()
        except Exception:
            logger.exception("Stage '%s' raised during stop().", stage.name)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the pipeline loop until ``request_stop()`` is called.

        Starts all stages, loops at ``_fps`` Hz, then stops all stages.
        ``on_shutdown`` is always called — even when ``start_stages()`` fails —
        so callers that acquire resources (overlay, sockets) in ``on_shutdown``
        cleanup code are guaranteed to run it.
        """
        self._stop_event.clear()
        stages_started = False

        try:
            if not self.start_stages():
                logger.error("PipelineEngine: failed to start stages; aborting.")
                return

            stages_started = True
            logger.info("PipelineEngine started (%d stages, %d fps).", len(self._stages), self._fps)
            self._loop()
        finally:
            if stages_started:
                self.stop_stages()
            if self._on_shutdown is not None:
                try:
                    self._on_shutdown()
                except Exception:
                    logger.exception("PipelineEngine: on_shutdown callback raised.")
            if stages_started:
                logger.info("PipelineEngine stopped.")

    def _loop(self) -> None:
        frame_id = 0
        while not self._stop_event.is_set():
            frame_id += 1
            t0 = time.monotonic()
            ctx = FrameContext(t0=t0, frame_id=frame_id)

            for stage in self._stages:
                with self._profiler.stage(stage.name):
                    try:
                        ctx = stage.process(ctx)
                    except Exception as exc:
                        err = PipelineStageError(stage.name, exc)
                        if self._error_policy == "halt":
                            raise err from exc
                        logger.exception(
                            "Stage '%s' raised in process(); continuing (error_policy=continue).",
                            stage.name,
                        )
                        # If a stage raised before setting capture_ok=True, mark the
                        # frame as failed so downstream stages skip gracefully.
                        if getattr(stage, "skip_on_capture_fail", False):
                            ctx.capture_ok = False

                # Short-circuit if this stage signalled no usable frame —
                # skip remaining stages rather than running gaze/gesture on
                # an empty context.  Uses a per-stage attribute instead of a
                # hard-coded stage-name comparison.
                if not ctx.capture_ok and getattr(stage, "skip_on_capture_fail", False):
                    # Sleep the remaining frame budget to avoid busy-spinning
                    # at ~200 Hz when the camera is temporarily unavailable.
                    elapsed = time.monotonic() - t0
                    budget = 1.0 / self._fps
                    remaining = budget - elapsed
                    time.sleep(max(remaining, 0.001))
                    break
            else:
                # All stages ran successfully.
                if self._on_frame is not None:
                    try:
                        self._on_frame(ctx)
                    except Exception:
                        logger.exception("PipelineEngine: on_frame callback raised.")

            self._profiler.tick()
            self._update_fps()

            # Pace to target FPS.
            elapsed = time.monotonic() - t0
            budget = 1.0 / self._fps
            sleep = budget - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _update_fps(self) -> None:
        self._fps_counter += 1
        elapsed = time.monotonic() - self._fps_timer
        if elapsed >= 1.0:
            self._actual_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer = time.monotonic()
            logger.debug("PipelineEngine FPS: %.1f", self._actual_fps)
