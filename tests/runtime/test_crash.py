"""Tests for runtime.crash — process-wide crash handlers."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import pytest

from gazecontrol.runtime.crash import install_crash_handlers, uninstall_crash_handlers


@pytest.fixture(autouse=True)
def _restore_handlers():
    yield
    uninstall_crash_handlers()


def test_thread_excepthook_logs_uncaught(caplog, tmp_path: Path):
    install_crash_handlers(fault_log_path=tmp_path / "fault.log")

    def boom():
        raise RuntimeError("background boom")

    with caplog.at_level(logging.CRITICAL, logger="gazecontrol.crash"):
        t = threading.Thread(target=boom, daemon=True)
        t.start()
        t.join(timeout=2.0)

    assert any("Uncaught exception on thread" in r.message for r in caplog.records)
    assert any("background boom" in (r.exc_text or "") for r in caplog.records)


def test_install_is_idempotent(tmp_path: Path):
    install_crash_handlers(fault_log_path=tmp_path / "fault.log")
    install_crash_handlers(fault_log_path=tmp_path / "fault.log")
    # Verify our hooks remain installed after a second call.
    assert sys.excepthook is not sys.__excepthook__
    assert threading.excepthook is not threading.__excepthook__


def test_signal_callback_called(monkeypatch, tmp_path: Path):
    """on_signal callback fires when handler invoked directly."""
    called = {"n": 0}

    def cb():
        called["n"] += 1

    install_crash_handlers(fault_log_path=tmp_path / "fault.log", on_signal=cb)

    from gazecontrol.runtime import crash

    crash._signal_handler(15, None)
    assert called["n"] == 1
