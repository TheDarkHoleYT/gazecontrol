"""Process-wide crash handlers.

Installs hooks so unexpected failures are captured to the log instead of
silently terminating the process:

- ``sys.excepthook``           — uncaught exceptions on the main thread.
- ``threading.excepthook``     — uncaught exceptions on background threads.
- ``faulthandler``             — C/C++ crashes (segfaults, ONNX/MediaPipe).
- POSIX ``SIGTERM`` / ``SIGINT`` — graceful shutdown via a registered callback.
- Qt ``qInstallMessageHandler`` — when PyQt6 is importable.

Idempotent: repeat calls overwrite the previous handlers.
"""

from __future__ import annotations

import faulthandler
import logging
import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import FrameType, TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger("gazecontrol.crash")

_fault_log_fp: object | None = None
_shutdown_cb: Callable[[], None] | None = None


def _excepthook(
    exc_type: type[BaseException],
    exc: BaseException,
    tb: TracebackType | None,
) -> None:
    """Log uncaught main-thread exceptions before the interpreter exits."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    logger.critical(
        "Uncaught exception on main thread",
        exc_info=(exc_type, exc, tb),
    )


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    """Log uncaught exceptions on background threads."""
    if issubclass(args.exc_type, SystemExit):
        return
    if args.exc_value is None:
        # Defensive: ExceptHookArgs.exc_value is typed Optional but set in
        # practice; skip rather than raise from inside the hook itself.
        return
    logger.critical(
        "Uncaught exception on thread %r",
        args.thread.name if args.thread is not None else "<unknown>",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _signal_handler(signum: int, _frame: FrameType | None) -> None:
    """Translate SIGTERM/SIGINT into a graceful shutdown callback."""
    name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    logger.info("Received %s; requesting shutdown.", name)
    if _shutdown_cb is not None:
        try:
            _shutdown_cb()
        except Exception:
            logger.exception("Shutdown callback raised.")


def install_crash_handlers(
    fault_log_path: Path | None = None,
    on_signal: Callable[[], None] | None = None,
) -> None:
    """Install all crash hooks.

    Args:
        fault_log_path: When set, ``faulthandler`` writes C-level tracebacks to
            this file (in append mode).  When ``None``, ``faulthandler`` writes
            to ``sys.stderr``.
        on_signal: Callable invoked when the process receives ``SIGTERM`` or
            (on Windows) ``SIGBREAK``.  Use it to set the pipeline stop event.
    """
    global _fault_log_fp, _shutdown_cb

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook

    # faulthandler — open the log lazily so callers can pass a per-run path.
    if fault_log_path is not None:
        try:
            fault_log_path.parent.mkdir(parents=True, exist_ok=True)
            _fault_log_fp = fault_log_path.open("a", buffering=1, encoding="utf-8")
            faulthandler.enable(file=_fault_log_fp, all_threads=True)
        except OSError:
            logger.exception("Could not open faulthandler log %s; using stderr.", fault_log_path)
            faulthandler.enable(all_threads=True)
    else:
        faulthandler.enable(all_threads=True)

    _shutdown_cb = on_signal

    # Signals — SIGINT is handled by Python's default to raise KeyboardInterrupt;
    # we wire SIGTERM and (on Windows) SIGBREAK so containerised / systemd
    # shutdowns drain the pipeline cleanly.
    if on_signal is not None:
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _signal_handler)
        if hasattr(signal, "SIGBREAK"):  # Windows-only
            signal.signal(signal.SIGBREAK, _signal_handler)  # type: ignore[attr-defined,unused-ignore]

    # Qt message handler — best-effort; available only when PyQt6 is loaded.
    try:
        from PyQt6.QtCore import QMessageLogContext, QtMsgType, qInstallMessageHandler

        def _qt_message_handler(
            msg_type: QtMsgType,
            _ctx: QMessageLogContext,
            message: str | None,
        ) -> None:
            level = {
                QtMsgType.QtDebugMsg: logging.DEBUG,
                QtMsgType.QtInfoMsg: logging.INFO,
                QtMsgType.QtWarningMsg: logging.WARNING,
                QtMsgType.QtCriticalMsg: logging.ERROR,
                QtMsgType.QtFatalMsg: logging.CRITICAL,
            }.get(msg_type, logging.WARNING)
            logger.log(level, "Qt: %s", message or "")

        qInstallMessageHandler(_qt_message_handler)
    except ImportError:
        # Qt not installed in the running environment (e.g. CI headless) — skip.
        pass


def uninstall_crash_handlers() -> None:
    """Restore stdlib defaults.  Test helper."""
    global _fault_log_fp, _shutdown_cb
    sys.excepthook = sys.__excepthook__
    threading.excepthook = threading.__excepthook__
    faulthandler.disable()
    if _fault_log_fp is not None:
        import contextlib

        with contextlib.suppress(OSError):
            _fault_log_fp.close()  # type: ignore[attr-defined]
        _fault_log_fp = None
    _shutdown_cb = None
