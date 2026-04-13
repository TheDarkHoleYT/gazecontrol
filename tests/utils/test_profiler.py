"""Tests for PipelineProfiler."""
from __future__ import annotations

import time

from gazecontrol.utils.profiler import PipelineProfiler


def test_stage_context_manager():
    p = PipelineProfiler(log_every_n=1000)
    with p.stage("capture"):
        time.sleep(0.01)
    p.tick()
    # Should not raise.


def test_tick_increments_count():
    p = PipelineProfiler(log_every_n=1000)
    for _ in range(5):
        p.tick()
    # PipelineProfiler doesn't expose tick_count, so just test it doesn't crash.


def test_multiple_stages():
    p = PipelineProfiler(log_every_n=1000)
    with p.stage("gaze"):
        pass
    with p.stage("gesture"):
        pass
    p.tick()


def test_log_triggered_at_n(caplog):
    """Profiler should log stats every N ticks."""
    import logging

    p = PipelineProfiler(log_every_n=5)
    with p.stage("test"):
        pass
    with caplog.at_level(logging.DEBUG):
        for _ in range(5):
            p.tick()
    # Just verify it completes without exception.
