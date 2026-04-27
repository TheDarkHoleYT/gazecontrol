"""GazeControl CLI entry point.

Configures logging (only here, not at import time), parses arguments,
shows the mode-selector dialog when needed, and launches the pipeline
in either HAND_ONLY or EYE_HAND mode.
"""

from __future__ import annotations

import argparse
import logging
import platform
import signal
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gazecontrol.runtime.input_mode import InputMode


def _suppress_third_party_logs() -> None:
    """Silence verbose C++ loggers from mediapipe/TF. Called inside main()."""
    import os

    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _detect_virtual_desktop() -> tuple[int, int, int, int]:
    """Return (left, top, width, height) of the virtual desktop.

    Uses per-monitor DPI awareness v2 for accurate coordinates on
    high-DPI / multi-monitor setups.
    """
    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDpiAwarenessContext(ctypes.c_ssize_t(-4))
        except (AttributeError, OSError):
            user32.SetProcessDPIAware()
        left = user32.GetSystemMetrics(76)
        top = user32.GetSystemMetrics(77)
        width = user32.GetSystemMetrics(78)
        height = user32.GetSystemMetrics(79)
        if width > 0 and height > 0:
            return (int(left), int(top), int(width), int(height))
    except (OSError, AttributeError) as exc:
        logging.getLogger(__name__).debug(
            "Virtual desktop probe failed; using 1920x1080 fallback: %s", exc
        )
    return (0, 0, 1920, 1080)


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------


def _resolve_mode(
    cli_mode: str | None,
    settings_mode: InputMode,
    *,
    show_dialog: bool,
) -> tuple[InputMode, bool]:
    """Decide which input mode to use and whether to show the dialog.

    Precedence: CLI > env (already in settings_mode) > persisted runtime.toml
    > settings default. Returns (mode, used_dialog).
    """
    from gazecontrol.runtime.input_mode import InputMode as _InputMode
    from gazecontrol.runtime.input_mode import load_persisted_mode

    if cli_mode:
        try:
            return _InputMode(cli_mode), False
        except ValueError as exc:
            raise SystemExit(f"Unknown --mode value: {cli_mode!r}") from exc

    persisted = load_persisted_mode()
    if persisted is not None and not show_dialog:
        return persisted, False

    if not show_dialog:
        return settings_mode, False

    initial = persisted or settings_mode
    chosen = _show_mode_dialog(initial)
    if chosen is None:
        return persisted or settings_mode, True
    return chosen, True


def _show_mode_dialog(initial: InputMode) -> InputMode | None:
    """Display the mode selector dialog. Returns the chosen InputMode or None."""
    try:
        from PyQt6.QtWidgets import QApplication

        from gazecontrol.overlay.mode_selector_dialog import ModeSelectorDialog
    except ImportError:
        logging.getLogger(__name__).warning("PyQt6 unavailable; skipping mode-selector dialog.")
        return None
    QApplication.instance() or QApplication(sys.argv)
    dialog = ModeSelectorDialog(initial=initial)
    dialog.exec()
    return dialog.result_mode


# ---------------------------------------------------------------------------
# CLI sub-commands
# ---------------------------------------------------------------------------


def _cmd_dump_config(*, resolved: bool = False) -> None:
    """Print resolved AppSettings as JSON and exit."""
    import json
    import os

    from gazecontrol.paths import Paths
    from gazecontrol.settings import get_settings

    s = get_settings()
    payload: dict[str, object] = {"settings": s.model_dump()}
    if resolved:
        payload["env"] = {k: v for k, v in os.environ.items() if k.startswith("GAZECONTROL_")}
        payload["paths"] = {
            "log_file": str(Paths.log_file()),
            "models": str(Paths.models()),
            "profiles": str(Paths.profiles()),
            "runtime_config": str(Paths.runtime_config()),
        }
    print(json.dumps(payload, indent=2, default=str))


def _doctor_rows() -> tuple[list[tuple[str, bool, str]], dict[str, bool]]:
    """Run probes and return (rows, machine-readable status dict)."""
    from gazecontrol.paths import Paths
    from gazecontrol.settings import get_settings

    s = get_settings()
    rows: list[tuple[str, bool, str]] = []
    status: dict[str, bool] = {}

    try:
        import cv2

        cap = cv2.VideoCapture(s.camera.index, cv2.CAP_DSHOW)
        cam_ok = cap.isOpened()
        if cam_ok:
            cap.release()
        rows.append(
            (
                f"Camera (index {s.camera.index})",
                cam_ok,
                "" if cam_ok else "Check connections / permissions",
            )
        )
        status["camera"] = cam_ok
    except (RuntimeError, OSError, cv2.error) as exc:
        rows.append(("Camera", False, str(exc)))
        status["camera"] = False

    hl_path = Paths.hand_landmarker()
    rows.append(
        (
            "Hand landmarker model",
            hl_path.exists(),
            "" if hl_path.exists() else f"Missing: {hl_path}",
        )
    )
    status["hand_landmarker_model"] = hl_path.exists()

    try:
        import eyetrax  # noqa: F401

        rows.append(("eyetrax (eye)", True, ""))
        status["eyetrax"] = True
    except ImportError:
        rows.append(("eyetrax (eye)", False, "Optional: pip install gazecontrol[eye]"))
        status["eyetrax"] = False

    l2cs_path = Paths.l2cs_model()
    rows.append(
        (
            "L2CS-Net model",
            l2cs_path.exists(),
            "" if l2cs_path.exists() else f"Optional: download to {l2cs_path}",
        )
    )
    status["l2cs_model"] = l2cs_path.exists()

    profile_path = Paths.gaze_profile(s.gaze.profile)
    rows.append(
        (
            f"Gaze profile '{s.gaze.profile}'",
            profile_path.exists(),
            "" if profile_path.exists() else "Run 'gazecontrol --calibrate-gaze'",
        )
    )
    status["gaze_profile"] = profile_path.exists()

    try:
        import PyQt6.QtWidgets  # noqa: F401

        rows.append(("PyQt6", True, ""))
        status["pyqt6"] = True
    except ImportError:
        rows.append(("PyQt6", False, "Overlay disabled — install PyQt6"))
        status["pyqt6"] = False

    return rows, status


def _cmd_doctor(*, as_json: bool = False) -> int:
    """Probe hardware and print a status table.

    When ``as_json`` is True a machine-readable JSON object is emitted
    instead of the unicode table; the exit code is unchanged.
    """
    rows, status = _doctor_rows()
    all_ok = all(ok or "Optional" in hint for _, ok, hint in rows)

    if as_json:
        import json

        print(json.dumps({"ok": all_ok, "checks": status}, indent=2))
        return 0 if all_ok else 1

    COL = 32
    print(f"\n{'─' * 60}")
    print("  gazecontrol --doctor")
    print(f"{'─' * 60}")
    for label, ok, hint in rows:
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {label:<{COL}} {hint}")
    print(f"{'─' * 60}\n")
    return 0 if all_ok else 1


def _cmd_healthcheck() -> int:
    """One-shot health probe: exit 0 only if camera + models are accessible.

    Maps probe failures to the exit codes defined in :mod:`gazecontrol.errors`
    so init systems and watchdogs can act on the specific failure class.
    """
    from gazecontrol.errors import EXIT_CAMERA, EXIT_MODEL_LOAD

    _, status = _doctor_rows()
    if not status.get("camera"):
        return EXIT_CAMERA
    if not status.get("hand_landmarker_model"):
        return EXIT_MODEL_LOAD
    return 0


def _cmd_benchmark(seconds: int, mode: InputMode) -> None:
    """Run the pipeline headless for *seconds* and print profiler percentiles."""
    import threading as _threading
    import time as _time

    from gazecontrol.runtime.pipeline_factory import PipelineFactory
    from gazecontrol.settings import get_settings

    vdesk = _detect_virtual_desktop()
    built = PipelineFactory(mode=mode, vdesk=vdesk, settings=get_settings()).build()
    engine = built.engine

    def _run() -> None:
        try:
            engine.run()
        except (RuntimeError, OSError) as exc:
            logging.getLogger(__name__).warning("Benchmark pipeline run() raised: %s", exc)

    t = _threading.Thread(target=_run, daemon=True)
    t.start()
    print(f"Running benchmark for {seconds}s (mode={mode.value})…", flush=True)
    _time.sleep(seconds)
    engine.request_stop()
    t.join(timeout=5.0)

    profiler = getattr(engine, "_profiler", None)
    if profiler is None:
        print("No profiler available.", file=sys.stderr)
        return
    pct = profiler.percentiles()
    if not pct:
        print("No profiler data collected.", file=sys.stderr)
        return
    total_mean = sum(v["mean"] for v in pct.values())
    COL = 16
    print(f"\n{'─' * 60}")
    print(f"  gazecontrol --benchmark {seconds}s")
    print(f"{'─' * 60}")
    print(f"  {'Stage':<{COL}} {'p50':>8} {'p95':>8} {'mean':>8}")
    print(f"  {'─' * COL} {'─' * 8} {'─' * 8} {'─' * 8}")
    for name, stats in pct.items():
        print(
            f"  {name:<{COL}} {stats['p50']:>7.1f}ms {stats['p95']:>7.1f}ms {stats['mean']:>7.1f}ms"
        )
    print(f"  {'─' * COL} {'─' * 8} {'─' * 8} {'─' * 8}")
    print(f"  {'TOTAL':<{COL}} {'':>8} {'':>8} {total_mean:>7.1f}ms")
    print(f"{'─' * 60}\n")


def _cmd_calibrate_gaze(profile: str) -> int:
    """Run the Qt-based gaze calibration UI."""
    from gazecontrol.calibration.runner import run_gaze_calibration

    vdesk = _detect_virtual_desktop()
    return run_gaze_calibration(profile=profile, vdesk=vdesk)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point — registered as ``gazecontrol`` console script."""
    from gazecontrol import __version__

    # SIGINT raises KeyboardInterrupt as usual; SIGTERM/SIGBREAK is wired by
    # install_crash_handlers() so a registered shutdown callback can drain the
    # pipeline cleanly.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    parser = argparse.ArgumentParser(
        prog="gazecontrol",
        description="Desktop control via hand gestures and (optional) eye tracking.",
    )
    parser.add_argument("--version", action="version", version=f"gazecontrol {__version__}")
    parser.add_argument(
        "--no-overlay", action="store_true", help="Disable HUD overlay (headless mode)."
    )
    parser.add_argument(
        "--mode",
        choices=["hand", "eye-hand"],
        default=None,
        help="Skip the selector and force an input mode.",
    )
    parser.add_argument(
        "--no-mode-selector",
        action="store_true",
        help="Skip the mode-selector dialog at startup.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--dump-config", action="store_true", help="Print resolved settings as JSON and exit."
    )
    parser.add_argument(
        "--resolved",
        action="store_true",
        help="With --dump-config: also include resolved env vars and paths.",
    )
    parser.add_argument(
        "--doctor", action="store_true", help="Probe camera, models, and dependencies."
    )
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="One-shot health probe (camera + models). Exit code matches error class.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --doctor: emit JSON instead of the unicode table.",
    )
    parser.add_argument(
        "--log-modules",
        default=None,
        help="Per-module log levels (e.g. gazecontrol.gaze:DEBUG,gazecontrol.gesture:INFO).",
    )
    parser.add_argument(
        "--benchmark",
        metavar="SECONDS",
        type=int,
        nargs="?",
        const=30,
        default=None,
        help="Run pipeline headless for N seconds and print latency percentiles.",
    )
    parser.add_argument(
        "--calibrate-gaze", action="store_true", help="Run the gaze calibration UI and exit."
    )
    parser.add_argument(
        "--profile", default=None, help="Calibration profile name (overrides GazeSettings.profile)."
    )
    args = parser.parse_args()

    if args.dump_config:
        _cmd_dump_config(resolved=args.resolved)
        return

    if args.healthcheck:
        sys.exit(_cmd_healthcheck())

    _suppress_third_party_logs()

    import os as _os

    from gazecontrol.logging_config import apply_module_levels, configure_logging, get_run_id
    from gazecontrol.paths import Paths
    from gazecontrol.settings import InputMode, LoggingSettings, get_settings

    s = get_settings()
    log_settings = s.logging
    env_level = _os.environ.get("GAZECONTROL_LOG_LEVEL")
    effective_level = args.log_level or env_level or s.logging.level
    if effective_level != s.logging.level:
        log_settings = LoggingSettings(
            level=effective_level,
            format=s.logging.format,
            rotation_mb=s.logging.rotation_mb,
            backup_count=s.logging.backup_count,
        )
    configure_logging(log_settings)
    if args.log_modules:
        apply_module_levels(args.log_modules)
    logger = logging.getLogger(__name__)

    from gazecontrol.runtime.crash import install_crash_handlers

    fault_path = Paths.log_file().parent / "faulthandler.log"
    install_crash_handlers(fault_log_path=fault_path)

    logger.info(
        "GazeControl %s starting — run_id=%s python=%s os=%s log=%s",
        __version__,
        get_run_id(),
        platform.python_version(),
        platform.platform(terse=True),
        Paths.log_file(),
    )

    if args.doctor:
        sys.exit(_cmd_doctor(as_json=args.json))

    if args.calibrate_gaze:
        profile = args.profile or s.gaze.profile
        sys.exit(_cmd_calibrate_gaze(profile))

    show_dialog = s.runtime.show_mode_selector and not args.no_mode_selector and args.mode is None
    mode, used_dialog = _resolve_mode(
        cli_mode=args.mode,
        settings_mode=s.runtime.input_mode,
        show_dialog=show_dialog,
    )
    if used_dialog and isinstance(mode, InputMode) and s.runtime.mode_selector_remember:
        from gazecontrol.runtime.input_mode import persist_mode

        persist_mode(mode)
    logger.info("GazeControl: input mode = %s", mode.value if hasattr(mode, "value") else mode)

    if args.benchmark is not None:
        _cmd_benchmark(args.benchmark, mode)
        return

    from gazecontrol.errors import GazeControlError, exit_code_for
    from gazecontrol.runtime.pipeline_factory import PipelineFactory

    vdesk = _detect_virtual_desktop()
    built = PipelineFactory(mode=mode, vdesk=vdesk, settings=s).build()
    engine = built.engine

    # Re-install crash handlers now that the engine exists so SIGTERM drains it.
    install_crash_handlers(fault_log_path=fault_path, on_signal=engine.request_stop)

    if args.no_overlay:
        try:
            engine.run()
        except KeyboardInterrupt:
            logger.info("Shutting down (interrupt).")
        except GazeControlError as exc:
            print(f"Error: {exc.user_message()}", file=sys.stderr)
            sys.exit(exit_code_for(exc))
        finally:
            engine.request_stop()
        return

    try:
        from PyQt6.QtWidgets import QApplication

        from gazecontrol.overlay.overlay_window import OverlayWindow
        from gazecontrol.pipeline.qt_adapter import QtPipelineThread
    except ImportError:
        logger.error("PyQt6 not available; use --no-overlay or install PyQt6.")
        sys.exit(1)

    app = QApplication.instance() or QApplication(sys.argv)
    overlay = OverlayWindow()
    overlay.create_widget()
    overlay.setup_launcher(built.launcher_apps, built.app_launcher)
    built.overlay_bridge_holder.append(overlay.toggle_launcher)

    qt_thread = QtPipelineThread(engine)
    qt_thread.frame_processed.connect(
        lambda ctx: overlay.update(
            fingertip_screen=ctx.pointer_screen or ctx.fingertip_screen,
            state=built.interaction_stage.state,
            hovered_window=ctx.hovered_window,
            gesture_id=ctx.gesture_label,
            gesture_confidence=ctx.gesture_confidence,
            interaction_kind=(ctx.interaction.kind.value if ctx.interaction else None),
            capture_ok=ctx.capture_ok,
            frame_bgr=ctx.frame_bgr,
            gaze_screen=ctx.gaze_screen,
            gaze_confidence=ctx.gaze_confidence,
            pointer_source=ctx.pointer_source,
            input_mode=mode.value if hasattr(mode, "value") else str(mode),
        )
    )

    def _on_pipeline_finished() -> None:
        logger.info("Pipeline finished; quitting Qt event loop.")
        app.quit()

    qt_thread.finished.connect(_on_pipeline_finished)
    qt_thread.start()

    exit_code = 0
    try:
        app.exec()
    except KeyboardInterrupt:
        logger.info("Shutting down (interrupt).")
    except GazeControlError as exc:
        print(f"Error: {exc.user_message()}", file=sys.stderr)
        exit_code = exit_code_for(exc)
    finally:
        overlay.stop()
        qt_thread.stop()
    if exit_code:
        sys.exit(exit_code)
