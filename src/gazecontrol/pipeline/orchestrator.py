"""GazeControlPipeline — top-level orchestrator.

Runs the pipeline loop in a QThread, emits frame_processed signals to the
overlay via Qt's QueuedConnection (thread-safe, no manual locking needed).

Threading model:
    Main thread     → Qt event loop (QApplication.exec())
    PipelineThread  → this class (_loop method, QThread)
    GrabberThread   → FrameGrabber._capture_loop (daemon thread, inside CaptureStage)

Stop is deterministic:
    requestInterruption() → loop exits → camera released → overlay stopped.
"""
from __future__ import annotations

import logging
import time

from gazecontrol.pipeline.capture_stage import CaptureStage
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.gaze_stage import GazeStage
from gazecontrol.pipeline.gesture_stage import GestureStage
from gazecontrol.pipeline.intent_stage import IntentStage
from gazecontrol.settings import get_settings
from gazecontrol.utils.profiler import PipelineProfiler
from gazecontrol.window_manager.windows_mgr import WindowsManager

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import QThread, pyqtSignal
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QThread = object  # type: ignore[misc,assignment]
    pyqtSignal = None  # type: ignore[assignment]


class GazeControlPipeline(QThread):  # type: ignore[misc]
    """Main pipeline — runs in a dedicated QThread, emits signals to the overlay.

    Instantiate, call ``start()``, and the Qt event loop handles the rest.
    Call ``stop()`` (or ``requestInterruption()``) to shut down cleanly.
    """

    #: Emitted each tick with a copy of the FrameContext for the HUD.
    frame_processed = pyqtSignal(object) if _QT_AVAILABLE and pyqtSignal else None  # type: ignore[misc]

    def __init__(
        self,
        profile_name: str = "default",
        screen_w: int = 1920,
        screen_h: int = 1080,
    ) -> None:
        if _QT_AVAILABLE:
            super().__init__()
        self.profile_name = profile_name
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._running = False

        s = get_settings()

        self._capture = CaptureStage()
        self._gaze = GazeStage(screen_w=screen_w, screen_h=screen_h)
        self._gesture = GestureStage(screen_w=screen_w, screen_h=screen_h)
        self._intent = IntentStage()
        self._window_manager = WindowsManager()
        self._profiler = PipelineProfiler(log_every_n=300)

        self._fps = s.camera.fps
        self._fps_counter = 0
        self._fps_timer = time.monotonic()
        self._actual_fps = 0.0

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Called by Qt when the thread starts (via QThread.start())."""
        self._running = True
        logger.info("GazeControlPipeline thread started.")
        self._init_estimator()
        self._loop()

    # ------------------------------------------------------------------

    def _init_estimator(self) -> None:
        """Initialise GazeEstimator in this thread (MediaPipe thread-safety)."""
        try:
            from eyetrax import GazeEstimator

            estimator = GazeEstimator(
                model_name="tiny_mlp",
                model_kwargs={
                    "hidden_layer_sizes": (256, 128, 64),
                    "max_iter": 1000,
                    "alpha": 1e-4,
                    "early_stopping": True,
                },
            )
            self._gaze.estimator = estimator
            logger.info("GazeEstimator (TinyMLP 256→128→64) initialized.")
            self._load_profile()
        except Exception:
            logger.exception("Failed to initialize GazeEstimator.")

    def _load_profile(self) -> bool:
        """Load calibration profile. Returns True on success."""
        from gazecontrol.paths import Paths

        profile_path = Paths.profiles() / f"{self.profile_name}.pkl"
        if not profile_path.exists():
            logger.warning(
                "Profile '%s' not found. Run --calibrate first.", self.profile_name
            )
            return False
        try:
            self._gaze.estimator.load_model(str(profile_path))  # type: ignore[union-attr]
            self._gaze.is_calibrated = True
            self._gaze._drift_corrector.reset()
            logger.info("Profile '%s' loaded.", self.profile_name)
            return True
        except Exception:
            logger.exception("Error loading profile '%s'.", self.profile_name)
            return False

    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main pipeline loop at ~FPS hz."""
        while self._running:
            if _QT_AVAILABLE and self.isInterruptionRequested():
                break

            t0 = time.monotonic()
            ctx = FrameContext(t0=t0)

            # Stage 1: Capture.
            with self._profiler.stage("capture"):
                ctx = self._capture.process(ctx)

            if not ctx.capture_ok:
                time.sleep(0.005)  # avoid busy-spin on camera drop
                continue

            # Stage 2: Gaze.
            with self._profiler.stage("gaze"):
                ctx = self._gaze.process(ctx)

            # Stage 3: Gesture.
            with self._profiler.stage("gesture"):
                ctx = self._gesture.process(ctx)

            # Stage 4: Intent.
            with self._profiler.stage("intent"):
                ctx = self._intent.process(ctx)

            # Stage 5: Window manager + drift feedback.
            action = ctx.action
            if action and isinstance(action, dict):
                if action.get("type") == "CLOSE_APP":
                    logger.info("Double pinch: shutting down GazeControl.")
                    self._running = False
                    break
                try:
                    self._window_manager.execute(action)
                except Exception:
                    logger.warning("WindowManager.execute failed.", exc_info=True)

                # Drift implicit recalibration.
                if (
                    ctx.target_window
                    and ctx.gaze_point
                    and action.get("type") in ("DRAG", "CLOSE", "MINIMIZE", "MAXIMIZE", "BRING_FRONT")
                ):
                    self._gaze.on_action(ctx.gaze_point, ctx.target_window)

            # Stage 6: Emit to overlay (via QueuedConnection — thread-safe).
            if _QT_AVAILABLE and self.frame_processed is not None:
                self.frame_processed.emit(ctx)

            self._profiler.tick()
            self._update_fps()

            # Pace the loop to target FPS (sleep only the remaining budget).
            loop_time = time.monotonic() - t0
            sleep = (1.0 / self._fps) - loop_time
            if sleep > 0:
                time.sleep(sleep)

        logger.info("GazeControlPipeline loop exited.")

    def _update_fps(self) -> None:
        self._fps_counter += 1
        elapsed = time.monotonic() - self._fps_timer
        if elapsed >= 1.0:
            self._actual_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer = time.monotonic()
            logger.debug("FPS: %.1f | State: %s", self._actual_fps, self._intent.state)

    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Request stop, join thread, release resources."""
        self._running = False
        if _QT_AVAILABLE:
            self.requestInterruption()
            self.wait(timeout=3000)  # ms; gives grabber/mediapipe time to flush
        self._capture.stop()
        self._gesture.close()
        logger.info("GazeControlPipeline stopped.")

    # ------------------------------------------------------------------
    # Non-Qt fallback: run without Qt (useful for testing / headless)
    # ------------------------------------------------------------------

    def run_blocking(self) -> None:
        """Run the loop in the calling thread (no Qt required)."""
        self._running = True
        self._init_estimator()
        self._loop()
