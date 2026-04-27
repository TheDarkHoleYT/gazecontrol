"""Tests for PipelineProfiler."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from gazecontrol.utils.profiler import PipelineProfiler


def test_stage_context_manager():
    p = PipelineProfiler(log_every_n=1000)
    with p.stage("capture"):
        time.sleep(0.01)
    p.tick()


def test_tick_increments_count():
    p = PipelineProfiler(log_every_n=1000)
    for _ in range(5):
        p.tick()
    assert p._tick_count == 5


def test_multiple_stages():
    p = PipelineProfiler(log_every_n=1000)
    with p.stage("gaze"):
        pass
    with p.stage("gesture"):
        pass
    p.tick()


def test_log_triggered_at_n(caplog):
    p = PipelineProfiler(log_every_n=5)
    with p.stage("test"):
        pass
    with caplog.at_level(logging.DEBUG):
        for _ in range(5):
            p.tick()


def test_stats_returns_mean():
    p = PipelineProfiler(log_every_n=1000)
    with p.stage("s"):
        time.sleep(0.01)
    s = p.stats()
    assert "s" in s
    assert s["s"] > 0.0


def test_percentiles_returns_p50_p95_mean():
    p = PipelineProfiler(log_every_n=1000)
    for _ in range(20):
        with p.stage("x"):
            time.sleep(0.001)
        p.tick()
    pct = p.percentiles()
    assert "x" in pct
    data = pct["x"]
    assert "p50" in data
    assert "p95" in data
    assert "mean" in data
    # p95 should be >= p50.
    assert data["p95"] >= data["p50"]


def test_emit_prometheus_writes_file(tmp_path: Path):
    p = PipelineProfiler(log_every_n=1000)
    with p.stage("capture"):
        time.sleep(0.005)
    p.tick()

    out = tmp_path / "metrics.prom"
    p.emit_prometheus(out)

    assert out.exists()
    content = out.read_text()
    assert "gazecontrol_stage_latency_ms" in content
    assert 'stage="capture"' in content
    assert "gazecontrol_tick_count" in content


def test_emit_prometheus_no_data_noop(tmp_path: Path):
    """emit_prometheus with no data should not create the file."""
    p = PipelineProfiler(log_every_n=1000)
    out = tmp_path / "empty.prom"
    p.emit_prometheus(out)
    assert not out.exists()


def test_budget_warning_after_streak(caplog):
    """Profiler should log a WARNING after consecutive over-budget frames."""
    p = PipelineProfiler(log_every_n=9999, fps_budget=1000)  # budget = 1 ms
    with caplog.at_level(logging.WARNING, logger="gazecontrol.utils.profiler"):
        for _ in range(6):
            with p.stage("slow"):
                time.sleep(0.005)  # 5 ms > 1 ms budget
            p.tick()
    assert any("over budget" in r.message.lower() for r in caplog.records)


def test_emit_prometheus_silent_on_bad_path():
    """emit_prometheus should not raise on an unwritable path."""
    p = PipelineProfiler(log_every_n=1000)
    with p.stage("x"):
        pass
    p.tick()
    # Non-existent directory — should log and return silently.
    p.emit_prometheus(Path("/does/not/exist/metrics.prom"))
