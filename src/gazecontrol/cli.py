"""GazeControl CLI entry point.

Configures logging (only here, not at import time), parses arguments,
and launches either calibration or the live pipeline.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

# Suppress verbose third-party loggers before importing heavy deps.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _setup_logging(level: str, log_file: str) -> None:
    """Configure root logger — called once from main(), never at import time."""
    from gazecontrol.paths import Paths

    log_path = Paths.log_file(log_file if log_file else None)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(str(log_path), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _detect_screen() -> tuple[int, int]:
    """Return (width, height) of the primary monitor with DPI awareness."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def main() -> None:
    """CLI entry point — registered as ``gazecontrol`` console script."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser(
        prog="gazecontrol",
        description="Desktop control via eye tracking and hand gestures.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run 5×5 grid calibration (25 points, ~50 s) with TinyMLP.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Run adaptive calibration (9+60 points, ~2.5 min, higher accuracy).",
    )
    parser.add_argument(
        "--profile",
        default="default",
        metavar="NAME",
        help="Calibration profile name (default: 'default').",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable HUD overlay (headless mode).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Override log file path (default: platformdirs user log dir).",
    )
    args = parser.parse_args()

    from gazecontrol.settings import get_settings

    s = get_settings()
    _setup_logging(args.log_level or s.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("GazeControl starting — profile=%s", args.profile)

    if args.calibrate or args.adaptive:
        from gazecontrol.pipeline.calibration import run_calibration

        run_calibration(profile_name=args.profile, adaptive=args.adaptive)
        return

    screen_w, screen_h = _detect_screen()

    if args.no_overlay:
        from gazecontrol.pipeline.orchestrator import GazeControlPipeline

        pipeline = GazeControlPipeline(
            profile_name=args.profile,
            screen_w=screen_w,
            screen_h=screen_h,
        )
        if not pipeline._capture.start():
            logger.error("Failed to open camera. Exiting.")
            sys.exit(1)
        try:
            pipeline.run_blocking()
        except KeyboardInterrupt:
            logger.info("User interrupt.")
        finally:
            pipeline.stop()
    else:
        from PyQt6.QtWidgets import QApplication

        from gazecontrol.overlay.overlay_window import OverlayWindow
        from gazecontrol.pipeline.orchestrator import GazeControlPipeline

        app = QApplication.instance() or QApplication(sys.argv)

        pipeline = GazeControlPipeline(
            profile_name=args.profile,
            screen_w=screen_w,
            screen_h=screen_h,
        )
        if not pipeline._capture.start():
            logger.error("Failed to open camera. Exiting.")
            sys.exit(1)

        overlay = OverlayWindow()
        overlay.create_widget()

        # Connect pipeline output → overlay via QueuedConnection (thread-safe).
        if pipeline.frame_processed is not None:
            pipeline.frame_processed.connect(  # type: ignore[union-attr]
                lambda ctx: overlay.update(
                    gaze_point=ctx.gaze_point,
                    state=pipeline._intent.state,
                    target_window=ctx.target_window,
                    gesture_id=ctx.gesture_label,
                    gesture_confidence=ctx.gesture_confidence,
                    is_calibrated=pipeline._gaze.is_calibrated,
                    gaze_event_type=ctx.fixation_event.type if ctx.fixation_event else None,
                )
            )

        # Stop signal from pipeline → quit Qt event loop.
        def _on_pipeline_finished() -> None:
            logger.info("Pipeline finished; quitting Qt event loop.")
            app.quit()

        if _QT_AVAILABLE:
            pipeline.finished.connect(_on_pipeline_finished)

        pipeline.start()

        try:
            app.exec()
        finally:
            pipeline.stop()
            overlay.stop()


try:
    from PyQt6.QtCore import QThread as _QT  # noqa: F401,N814
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
