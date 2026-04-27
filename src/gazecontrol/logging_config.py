"""Centralised logging configuration for GazeControl.

Call ``configure_logging(settings.logging)`` once at startup (inside
``main()``), never at import time.

JSON logging requires the optional extra::

    pip install gazecontrol[logging]
"""

from __future__ import annotations

import logging
import logging.config
import logging.handlers
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gazecontrol.settings import LoggingSettings

_DEFAULT_FMT = (
    "%(asctime)s %(levelname)-8s [%(run_id)s] frame=%(frame_id)d "
    "%(threadName)s %(name)s: %(message)s"
)
_configured = False
_run_id: str = uuid.uuid4().hex[:8]


class _RunIdFilter(logging.Filter):
    """Injects a short run-identifier into every log record."""

    def __init__(self) -> None:
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id
        return True


def get_run_id() -> str:
    """Return the current run identifier (8-char hex, stable within a process)."""
    return _run_id


def reset_logging() -> None:
    """Reset the configured flag so handlers are re-installed on next call.

    Intended for use in tests only.
    """
    global _configured
    _configured = False


def configure_logging(settings: LoggingSettings) -> None:
    """Apply *settings* to the root logger via ``dictConfig``.

    Creates:
    - A ``RotatingFileHandler`` writing to ``Paths.log_file()``.
    - A ``StreamHandler`` writing to stdout.

    Every log record receives a ``run_id`` field (8-char hex) and
    ``threadName`` so concurrent stage logs can be correlated.

    Args:
        settings: ``LoggingSettings`` instance (from ``AppSettings.logging``).

    Raises:
        ImportError: When ``settings.format == "json"`` but
            ``python-json-logger`` is not installed. The message includes the
            install command so the user knows how to fix it.
    """
    global _configured
    # Remove existing root handlers before re-configuring to prevent duplicate
    # log lines when configure_logging is called more than once (e.g. in tests).
    root = logging.getLogger()
    for handler in root.handlers[:]:
        try:
            handler.close()
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).debug("Could not close existing log handler: %s", exc)
        root.removeHandler(handler)

    from gazecontrol.paths import Paths

    log_path = Paths.log_file()

    if settings.format == "json":
        try:
            import pythonjsonlogger.jsonlogger
        except ImportError as exc:
            raise ImportError(
                "JSON logging requires the 'python-json-logger' package. "
                "Install it with:  pip install gazecontrol[logging]"
            ) from exc

        import socket

        from gazecontrol import __version__

        class _JsonFormatter(pythonjsonlogger.jsonlogger.JsonFormatter):  # type: ignore[misc]
            def add_fields(
                self,
                log_record: dict[str, Any],
                record: logging.LogRecord,
                message_dict: dict[str, Any],
            ) -> None:
                super().add_fields(log_record, record, message_dict)
                log_record["service"] = "gazecontrol"
                log_record["version"] = __version__
                log_record["host"] = socket.gethostname()
                log_record["run_id"] = _run_id

        formatter_config: dict[str, Any] = {
            "default": {
                "()": _JsonFormatter,
                "format": "%(asctime)s %(levelname)s %(name)s %(threadName)s %(message)s",
            }
        }
    else:
        formatter_config = {
            "default": {
                "format": _DEFAULT_FMT,
            }
        }

    run_id_filter = {
        "()": "gazecontrol.logging_config._RunIdFilter",
    }
    correlation_filter = {
        "()": "gazecontrol.utils.correlation.CorrelationFilter",
    }

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "run_id": run_id_filter,
            "correlation": correlation_filter,
        },
        "formatters": formatter_config,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
                "filters": ["run_id", "correlation"],
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default",
                "filename": str(log_path),
                "maxBytes": settings.rotation_mb * 1024 * 1024,
                "backupCount": settings.backup_count,
                "encoding": "utf-8",
                "filters": ["run_id", "correlation"],
            },
        },
        "root": {
            "level": settings.level,
            "handlers": ["console", "file"],
        },
    }
    logging.config.dictConfig(config)


def apply_module_levels(spec: str) -> None:
    """Apply per-module log levels from a CLI/env spec.

    Spec format: ``"gazecontrol.gaze:DEBUG,gazecontrol.gesture:INFO"``.
    Whitespace around tokens is tolerated; unknown levels are skipped with
    a warning.
    """
    if not spec:
        return
    log = logging.getLogger(__name__)
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            log.warning("Bad --log-modules entry %r (expected NAME:LEVEL).", entry)
            continue
        name, level_str = (s.strip() for s in entry.split(":", 1))
        level = logging.getLevelName(level_str.upper())
        if not isinstance(level, int):
            log.warning("Unknown log level %r for module %r.", level_str, name)
            continue
        logging.getLogger(name).setLevel(level)
        log.debug("Set %s logger to %s", name, level_str.upper())
