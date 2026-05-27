"""GazeControl CLI entry point.

Configures logging (only here, not at import time), parses arguments,
shows the mode-selector dialog when needed, and launches the pipeline
in either HAND_ONLY or EYE_HAND mode.
"""

from __future__ import annotations

import argparse
import contextlib
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


def _doctor_rows(
    *,
    functional: bool = False,
) -> tuple[list[tuple[str, bool, str]], dict[str, bool]]:
    """Run probes and return (rows, machine-readable status dict).

    When *functional* is True the doctor additionally runs the models
    on a dummy frame to catch installs that have the files on disk but
    cannot actually invoke them (broken DirectML, missing CUDA, etc.).
    The functional probes are time-budgeted so a hung model never
    blocks the CLI for more than ~5 s.
    """
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

    if functional:
        _doctor_functional_probes(rows, status)

    return rows, status


def _doctor_functional_probes(
    rows: list[tuple[str, bool, str]],
    status: dict[str, bool],
) -> None:
    """Live inference probes for ``--doctor --functional`` (G13).

    Each probe runs against a dummy in-memory frame and is wrapped in
    a broad except so a single broken backend cannot mask the rest of
    the report. Latency is recorded inline in the row's hint column.
    """
    import time as _time

    import numpy as _np

    from gazecontrol.paths import Paths

    # Capture one frame from the camera (timed).
    try:
        import cv2

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        t0 = _time.perf_counter()
        ok = cap.isOpened()
        frame_ok = False
        if ok:
            ok2, _frame = cap.read()
            cap.release()
            frame_ok = bool(ok2 and _frame is not None)
        elapsed_ms = (_time.perf_counter() - t0) * 1000.0
        rows.append(
            (
                "Camera frame capture",
                frame_ok,
                f"{elapsed_ms:.0f} ms" if frame_ok else "no frame received",
            )
        )
        status["camera_frame"] = frame_ok
    except Exception as exc:
        rows.append(("Camera frame capture", False, str(exc)))
        status["camera_frame"] = False

    # Dummy MediaPipe Face Landmarker probe.
    fl_ok = False
    fl_hint = ""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        lm_path = Paths.face_landmarker()
        if not lm_path.exists():
            fl_hint = f"Missing: {lm_path}"
        else:
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(lm_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
            )
            landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
            dummy = _np.zeros((224, 224, 3), dtype=_np.uint8)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy)
            t0 = _time.perf_counter()
            landmarker.detect(mp_img)
            elapsed_ms = (_time.perf_counter() - t0) * 1000.0
            landmarker.close()
            fl_ok = True
            fl_hint = f"{elapsed_ms:.0f} ms"
    except Exception as exc:
        fl_hint = str(exc)
    rows.append(("Face Landmarker inference", fl_ok, fl_hint))
    status["face_landmarker_inference"] = fl_ok

    # Dummy L2CS-Net ONNX probe — checks that the ONNX session loads
    # and ``predict`` returns without raising. Black dummy crops legally
    # return None (no face) so we treat anything-but-exception as OK.
    l2cs_ok = False
    l2cs_hint = ""
    try:
        from gazecontrol.gaze.l2cs_model import L2CSModel

        model_path = Paths.l2cs_model()
        if not model_path.exists():
            l2cs_hint = f"Missing: {model_path}"
        else:
            model = L2CSModel(str(model_path))
            if not model.is_loaded:
                l2cs_hint = "ONNX session refused to initialise"
            else:
                bgr_dummy = _np.zeros((224, 224, 3), dtype=_np.uint8)
                t0 = _time.perf_counter()
                _angles = model.predict(bgr_dummy)
                elapsed_ms = (_time.perf_counter() - t0) * 1000.0
                l2cs_ok = True
                if _angles is None:
                    l2cs_hint = f"{elapsed_ms:.0f} ms (no face — expected for dummy)"
                else:
                    l2cs_hint = f"{elapsed_ms:.0f} ms (yaw={_angles[0]:.1f}°)"
    except Exception as exc:
        l2cs_hint = str(exc)
    rows.append(("L2CS-Net inference", l2cs_ok, l2cs_hint))
    status["l2cs_inference"] = l2cs_ok

    # Calibration staleness probe — compute mean Euclidean error of the
    # cached training set against the current mapper. A large value
    # suggests the user has moved since the last calibration.
    try:
        from gazecontrol.gaze.gaze_mapper import GazeMapper
        from gazecontrol.settings import get_settings

        s = get_settings()
        mapper = GazeMapper()
        v2_path = Paths.resolve_active_v2_profile(s.gaze.user_id)
        legacy_path = Paths.gaze_profile(s.gaze.profile)
        load_target = v2_path or (legacy_path if legacy_path.exists() else None)
        if load_target is None:
            rows.append(("Calibration freshness", False, "No profile loaded"))
            status["calibration_freshness"] = False
        elif not mapper.load(load_target):
            rows.append(("Calibration freshness", False, f"Load failed: {load_target}"))
            status["calibration_freshness"] = False
        else:
            train_angles = mapper._training_angles
            train_targets = mapper._training_targets
            if train_angles is None or train_targets is None:
                rows.append(
                    (
                        "Calibration freshness",
                        True,
                        "Legacy v1 profile — recalibrate for richer telemetry",
                    )
                )
                status["calibration_freshness"] = True
            else:
                err = mapper.compute_holdout_error(train_angles, train_targets)
                fresh = err is not None and err < 80.0
                rows.append(
                    (
                        "Calibration freshness",
                        fresh,
                        f"recall error {err:.1f} px" if err is not None else "n/a",
                    )
                )
                status["calibration_freshness"] = bool(fresh)
    except Exception as exc:
        rows.append(("Calibration freshness", False, str(exc)))
        status["calibration_freshness"] = False


def _cmd_doctor(*, as_json: bool = False, functional: bool = False) -> int:
    """Probe hardware and print a status table.

    When ``as_json`` is True a machine-readable JSON object is emitted
    instead of the unicode table; the exit code is unchanged.
    When *functional* is True the doctor also runs live inference on
    dummy frames (G13) — useful for catching DirectML / CUDA setups
    that resolve the model file but fail at ORT init time.
    """
    rows, status = _doctor_rows(functional=functional)
    all_ok = all(ok or "Optional" in hint for _, ok, hint in rows)

    if as_json:
        import json

        print(json.dumps({"ok": all_ok, "checks": status}, indent=2))
        return 0 if all_ok else 1

    COL = 32
    print(f"\n{'─' * 60}")
    header = "  gazecontrol --doctor"
    if functional:
        header += " --functional"
    print(header)
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


def _cmd_benchmark(
    seconds: int,
    mode: InputMode,
    *,
    bench_json: str | None = None,
    mock_camera: bool = False,
) -> int:
    """Run the pipeline headless for *seconds* and print profiler percentiles.

    When *bench_json* is set the percentile snapshot is also written to
    the path as a JSON object (G11) so CI can gate on it::

        {
          "ok": true,
          "seconds": 20,
          "mode": "hand",
          "stages": {"capture": {"p50":..., "p95":..., "mean":...}, ...},
          "total_mean_ms": <sum of stage means>,
          "p95_total_ms": <max stage p95>,
          "sla_p95_ms": <env GAZECONTROL_SLA_P95_MS or 33.3>
        }

    Returns the process exit code: 0 when ``p95_total_ms`` is below the
    SLA budget (or no SLA is configured), 1 otherwise.
    """
    import json as _json
    import os as _os
    import threading as _threading
    import time as _time

    from gazecontrol.runtime.pipeline_factory import PipelineFactory
    from gazecontrol.settings import get_settings

    # G11 — CI flow: patch cv2.VideoCapture with a synthetic source so
    # headless runners without a webcam can still produce the
    # percentile snapshot the SLA gate reads.
    if mock_camera:
        from gazecontrol.utils.synthetic_capture import install_synthetic_capture

        install_synthetic_capture()

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
        return 1
    pct = profiler.percentiles()
    if not pct:
        print("No profiler data collected.", file=sys.stderr)
        return 1
    total_mean = sum(v["mean"] for v in pct.values())
    # G11 — pick the worst-stage p95 as the SLA metric. Stage p95s never
    # overlap because the pipeline runs sequentially, so max(stage_p95)
    # ≤ sum(stage_p95) and is the closest single number to a "tail
    # latency budget per frame".
    p95_total = max(v["p95"] for v in pct.values())
    sla_budget = float(_os.environ.get("GAZECONTROL_SLA_P95_MS", "33.3"))
    ok = p95_total <= sla_budget

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
    print(
        f"  p95 (worst stage) = {p95_total:.1f} ms  "
        f"[SLA ≤ {sla_budget:.1f} ms → {'OK' if ok else 'FAIL'}]"
    )
    print(f"{'─' * 60}\n")

    if bench_json is not None:
        payload = {
            "ok": ok,
            "seconds": seconds,
            "mode": mode.value if hasattr(mode, "value") else str(mode),
            "stages": {
                name: {
                    "p50": float(stats["p50"]),
                    "p95": float(stats["p95"]),
                    "mean": float(stats["mean"]),
                }
                for name, stats in pct.items()
            },
            "total_mean_ms": float(total_mean),
            "p95_total_ms": float(p95_total),
            "sla_p95_ms": float(sla_budget),
        }
        try:
            from pathlib import Path

            Path(bench_json).write_text(
                _json.dumps(payload, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            print(f"Failed to write bench JSON to {bench_json}: {exc}", file=sys.stderr)
            return 1

    return 0 if ok else 1


def _cmd_calibrate_gaze(
    profile: str,
    *,
    subset_size: int = 13,
    base_profile: str | None = None,
) -> int:
    """Run the Qt-based gaze calibration UI."""
    from gazecontrol.calibration.runner import run_gaze_calibration

    vdesk = _detect_virtual_desktop()
    return run_gaze_calibration(
        profile=profile,
        vdesk=vdesk,
        subset_size=subset_size,
        base_profile=base_profile,
    )


def _cmd_purge_profiles(*, assume_yes: bool, as_json: bool) -> int:
    """Erase every locally-stored gaze profile + the runtime config (G16).

    Implements the GDPR Art.17 right-to-erasure documented in PRIVACY.md.
    Prompts for confirmation unless ``--yes`` is passed; the prompt is
    *not* shown when stdin is not a TTY (CI / scripts must explicitly
    opt in with ``--yes``).

    Always logs a ``compliance.purge`` event so the user-owned audit
    trail (the log file itself) records the action.
    """
    import json as _json
    import shutil

    from gazecontrol.paths import Paths

    profiles_dir = Paths.profiles()
    runtime_cfg = Paths.runtime_config()
    targets = {
        "profiles_dir": str(profiles_dir),
        "runtime_config": str(runtime_cfg),
    }

    if not assume_yes:
        if not sys.stdin.isatty():
            print(
                "--purge-profiles refuses to delete without --yes when stdin is "
                "not a TTY. Add --yes to opt in.",
                file=sys.stderr,
            )
            return 2
        print("This will permanently delete:")
        for label, path in targets.items():
            print(f"  - {label}: {path}")
        try:
            reply = input("Type 'yes' to confirm: ").strip().lower()
        except EOFError:
            reply = ""
        if reply != "yes":
            print("Aborted — nothing deleted.", file=sys.stderr)
            return 1

    deleted: dict[str, bool] = {"profiles_dir": False, "runtime_config": False}
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir, ignore_errors=True)
        deleted["profiles_dir"] = not profiles_dir.exists()
    if runtime_cfg.exists():
        try:
            runtime_cfg.unlink()
            deleted["runtime_config"] = True
        except OSError:
            deleted["runtime_config"] = False

    logging.getLogger("gazecontrol.compliance").info(
        "compliance.purge",
        extra={
            "profiles_dir": str(profiles_dir),
            "runtime_config": str(runtime_cfg),
            "deleted": deleted,
        },
    )

    if as_json:
        print(_json.dumps({"deleted": deleted, "paths": targets}, indent=2))
    else:
        for label, ok in deleted.items():
            icon = "✓" if ok else "·"
            print(f"  {icon}  {label}: {targets[label]}")
        print("Done.")
    return 0


def _cmd_migrate_profiles(*, dry_run: bool, as_json: bool) -> int:
    """Migrate pre-v1.0 flat gaze profiles into the v2 layout (ADR-0009).

    Returns the process exit code: 0 on success (including "nothing to
    migrate"), 1 when one or more migrations errored.
    """
    import json as _json

    from gazecontrol.runtime.profile_migrate import migrate_profiles

    results = migrate_profiles(dry_run=dry_run)
    error_count = sum(1 for r in results if r.action == "error")

    if as_json:
        payload = [
            {
                "source": str(r.source),
                "target": str(r.target),
                "action": r.action,
                "message": r.message,
            }
            for r in results
        ]
        print(_json.dumps({"results": payload, "errors": error_count}, indent=2))
    else:
        if not results:
            print("No legacy gaze profiles found — nothing to migrate.")
        else:
            print(f"Found {len(results)} legacy profile(s):")
            for r in results:
                print(f"  [{r.action:8}] {r.source} → {r.target}")
                if r.message:
                    print(f"           {r.message}")
        if dry_run:
            print("\n(dry run — no files were copied)")
    return 1 if error_count else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point — registered as ``gazecontrol`` console script."""
    from gazecontrol import __version__

    # Force utf-8 on stdout/stderr so unicode glyphs in tables, dashes, and
    # log messages don't trigger UnicodeEncodeError on Windows cp1252 consoles.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(AttributeError, OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")

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
        "--functional",
        action="store_true",
        help=(
            "With --doctor: also run live inference probes against dummy frames "
            "(G13). Surfaces DirectML / CUDA / ORT init failures that the "
            "existence-only checks would miss."
        ),
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
        "--bench-json",
        metavar="PATH",
        default=None,
        help=(
            "With --benchmark: also dump the percentile snapshot as JSON "
            "(G11). The exit code reflects the worst-stage p95 against "
            "the SLA budget (env GAZECONTROL_SLA_P95_MS, default 33.3 ms)."
        ),
    )
    parser.add_argument(
        "--bench-mock",
        action="store_true",
        help=(
            "With --benchmark: replace cv2.VideoCapture with a synthetic "
            "frame source. Lets headless CI runners exercise the "
            "pipeline + measure stage latency without a webcam (G11)."
        ),
    )
    parser.add_argument(
        "--calibrate-gaze", action="store_true", help="Run the gaze calibration UI and exit."
    )
    parser.add_argument(
        "--profile", default=None, help="Calibration profile name (overrides GazeSettings.profile)."
    )
    parser.add_argument(
        "--calibrate-incremental",
        metavar="N",
        type=int,
        choices=[3, 5, 9, 13],
        default=None,
        help=(
            "Run an incremental top-up calibration (G8): captures only N "
            "targets (3, 5, 9, or 13) and partial-fits the existing profile "
            "instead of re-running the full grid. Requires --profile to "
            "name an existing profile."
        ),
    )
    parser.add_argument(
        "--migrate-profiles",
        action="store_true",
        help=(
            "Migrate pre-v1.0 flat *.gaze.npz profiles into the v2 "
            "<user>/<monitor>/v{N}.npz layout (ADR-0009) and exit. "
            "Originals are preserved."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --migrate-profiles: report planned actions without writing.",
    )
    parser.add_argument(
        "--purge-profiles",
        action="store_true",
        help=(
            "Erase every locally-stored gaze profile + runtime.toml and "
            "exit (G16, GDPR Art.17). Prompts for confirmation unless "
            "--yes is passed. See PRIVACY.md for the data inventory."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="With --purge-profiles: skip the interactive confirmation prompt.",
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
        sys.exit(_cmd_doctor(as_json=args.json, functional=args.functional))

    if args.migrate_profiles:
        sys.exit(_cmd_migrate_profiles(dry_run=args.dry_run, as_json=args.json))

    if args.purge_profiles:
        sys.exit(_cmd_purge_profiles(assume_yes=args.yes, as_json=args.json))

    if args.calibrate_gaze:
        profile = args.profile or s.gaze.profile
        sys.exit(_cmd_calibrate_gaze(profile))

    if args.calibrate_incremental is not None:
        # Incremental flow needs an existing profile to top up.
        if not args.profile:
            print(
                "--calibrate-incremental requires --profile NAME of an existing profile.",
                file=sys.stderr,
            )
            sys.exit(2)
        sys.exit(
            _cmd_calibrate_gaze(
                args.profile,
                subset_size=args.calibrate_incremental,
                base_profile=args.profile,
            )
        )

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
        sys.exit(
            _cmd_benchmark(
                args.benchmark,
                mode,
                bench_json=args.bench_json,
                mock_camera=args.bench_mock,
            )
        )

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
