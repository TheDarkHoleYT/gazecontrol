"""Tests for configure_logging — handler creation, file path, JSON opt-in."""

from __future__ import annotations

import contextlib
import logging
import logging.handlers

import pytest

from gazecontrol.logging_config import configure_logging
from gazecontrol.settings import LoggingSettings


def _reset_root_logger():
    """Remove all handlers from the root logger (test isolation)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        with contextlib.suppress(Exception):
            h.close()


@pytest.fixture(autouse=True)
def _clean_handlers():
    """Ensure root logger handlers are clean before and after each test."""
    _reset_root_logger()
    yield
    _reset_root_logger()


def test_configure_logging_creates_handlers(tmp_path, monkeypatch):
    """configure_logging should attach a StreamHandler and RotatingFileHandler."""
    from gazecontrol.paths import Paths

    log_file = tmp_path / "gazecontrol.log"
    monkeypatch.setattr(Paths, "log_file", staticmethod(lambda override=None: log_file))

    configure_logging(LoggingSettings(level="DEBUG"))

    root = logging.getLogger()
    handler_types = {type(h) for h in root.handlers}
    assert logging.StreamHandler in handler_types
    assert logging.handlers.RotatingFileHandler in handler_types


def test_configure_logging_log_file_created(tmp_path, monkeypatch):
    from gazecontrol.paths import Paths

    log_file = tmp_path / "gazecontrol.log"
    monkeypatch.setattr(Paths, "log_file", staticmethod(lambda override=None: log_file))

    configure_logging(LoggingSettings())

    assert log_file.exists()


def test_configure_logging_respects_level(tmp_path, monkeypatch):
    from gazecontrol.paths import Paths

    log_file = tmp_path / "gazecontrol.log"
    monkeypatch.setattr(Paths, "log_file", staticmethod(lambda override=None: log_file))

    configure_logging(LoggingSettings(level="WARNING"))

    assert logging.getLogger().level == logging.WARNING


def test_configure_logging_json_missing_package_raises(tmp_path, monkeypatch):
    """format='json' without python-json-logger must raise ImportError with guidance."""
    import sys

    from gazecontrol.paths import Paths

    log_file = tmp_path / "gazecontrol.log"
    monkeypatch.setattr(Paths, "log_file", staticmethod(lambda override=None: log_file))

    # Simulate the package being absent by blocking the import.
    original = sys.modules.get("pythonjsonlogger")
    sys.modules["pythonjsonlogger"] = None  # type: ignore[assignment]

    try:
        with pytest.raises(ImportError, match="gazecontrol\\[logging\\]"):
            configure_logging(LoggingSettings(format="json"))
    finally:
        if original is None:
            sys.modules.pop("pythonjsonlogger", None)
        else:
            sys.modules["pythonjsonlogger"] = original
