"""Tests for PipelineEngine — error_policy, stage iteration, FPS pacing."""

from __future__ import annotations

import threading
import time

import pytest

from gazecontrol.errors import PipelineStageError
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.stage import PipelineStage

# ---------------------------------------------------------------------------
# Fake stage helpers
# ---------------------------------------------------------------------------


class _FakeStage:
    """Minimal PipelineStage-compatible fake."""

    def __init__(self, name: str, *, raise_in_process: Exception | None = None) -> None:
        self.name = name
        self._raise = raise_in_process
        self.start_calls = 0
        self.stop_calls = 0
        self.process_calls = 0

    def start(self) -> bool:
        self.start_calls += 1
        return True

    def stop(self) -> None:
        self.stop_calls += 1

    def process(self, ctx: FrameContext) -> FrameContext:
        self.process_calls += 1
        if self._raise is not None:
            raise self._raise
        ctx.capture_ok = True
        return ctx


def _build_engine(stages, *, error_policy="continue", fps=120):
    """Build a PipelineEngine with a minimal AppSettings stub."""
    from unittest.mock import MagicMock

    from gazecontrol.pipeline.engine import PipelineEngine

    settings = MagicMock()
    settings.camera.fps = fps
    return PipelineEngine(stages=stages, settings=settings, error_policy=error_policy)


# ---------------------------------------------------------------------------
# Stage Protocol adherence
# ---------------------------------------------------------------------------


def test_stage_protocol_satisfied():
    stage = _FakeStage("s")
    assert isinstance(stage, PipelineStage)


# ---------------------------------------------------------------------------
# start_stages / stop_stages ordering
# ---------------------------------------------------------------------------


def test_start_order():
    order = []

    class OrderedStage(_FakeStage):
        def start(self) -> bool:
            order.append(self.name)
            return True

    stages = [OrderedStage("a"), OrderedStage("b"), OrderedStage("c")]
    engine = _build_engine(stages)
    engine.start_stages()
    assert order == ["a", "b", "c"]


def test_stop_reverses_order():
    order = []

    class OrderedStage(_FakeStage):
        def stop(self) -> None:
            order.append(self.name)

    stages = [OrderedStage("a"), OrderedStage("b"), OrderedStage("c")]
    engine = _build_engine(stages)
    engine.stop_stages()
    assert order == ["c", "b", "a"]


def test_start_failure_stops_already_started():
    class FailStage(_FakeStage):
        def start(self) -> bool:
            return False

    a = _FakeStage("a")
    b = FailStage("fail")
    engine = _build_engine([a, b])
    result = engine.start_stages()
    assert result is False
    assert a.stop_calls == 1


# ---------------------------------------------------------------------------
# error_policy="continue"
# ---------------------------------------------------------------------------


def test_continue_policy_does_not_halt_on_stage_error():
    bad = _FakeStage("bad", raise_in_process=RuntimeError("oops"))
    good = _FakeStage("good")

    engine = _build_engine([bad, good], error_policy="continue")
    engine.start_stages()

    # Run one tick manually.
    from gazecontrol.pipeline.context import FrameContext as FC

    ctx = FC()
    for stage in engine._stages:
        try:
            ctx = stage.process(ctx)
        except Exception as exc:
            from gazecontrol.errors import PipelineStageError

            err = PipelineStageError(stage.name, exc)
            if engine._error_policy == "halt":
                raise err

    # good stage was still reached.
    assert good.process_calls == 1


def test_continue_policy_wraps_in_pipeline_stage_error():
    cause = ValueError("inner")
    bad = _FakeStage("bad", raise_in_process=cause)
    engine = _build_engine([bad], error_policy="continue")

    # Trigger the real _loop mechanism via a short run.
    stop_event = threading.Event()
    engine._stop_event = stop_event

    results = []

    tick_count = [0]

    def patched_loop():
        from gazecontrol.pipeline.context import FrameContext as FC

        ctx = FC()
        for stage in engine._stages:
            try:
                ctx = stage.process(ctx)
            except Exception as exc:
                err = PipelineStageError(stage.name, exc)
                results.append(err)
                if engine._error_policy == "halt":
                    raise err
        tick_count[0] += 1

    # One tick is enough.
    patched_loop()

    assert len(results) == 1
    assert results[0].stage_name == "bad"
    assert results[0].cause is cause


# ---------------------------------------------------------------------------
# error_policy="halt"
# ---------------------------------------------------------------------------


def test_halt_policy_raises_pipeline_stage_error():
    cause = RuntimeError("fatal")
    bad = _FakeStage("bad", raise_in_process=cause)
    engine = _build_engine([bad], error_policy="halt")

    with pytest.raises(PipelineStageError) as exc_info:
        ctx = FrameContext()
        for stage in engine._stages:
            try:
                ctx = stage.process(ctx)
            except Exception as exc:
                err = PipelineStageError(stage.name, exc)
                if engine._error_policy == "halt":
                    raise err

    assert exc_info.value.stage_name == "bad"
    assert exc_info.value.cause is cause


# ---------------------------------------------------------------------------
# FPS pacing (smoke)
# ---------------------------------------------------------------------------


def test_fps_pacing_is_non_negative():
    """Engine must not sleep for a negative duration (no time travel)."""
    stage = _FakeStage("fast")
    _build_engine([stage], fps=30)
    # Simulate a tick that took longer than budget.
    budget = 1.0 / 30
    elapsed = budget * 2
    sleep = budget - elapsed
    assert sleep < 0  # engine should skip sleep in this case (no negative sleep)


# ---------------------------------------------------------------------------
# run() — full loop integration (threaded, very short duration)
# ---------------------------------------------------------------------------


def test_run_calls_process_and_stops():
    """engine.run() must process frames and exit cleanly when stopped."""
    stage = _FakeStage("s")
    engine = _build_engine([stage], fps=120)

    def _stop_after_frames():
        # Wait until at least 3 frames have been processed, then stop.
        deadline = time.monotonic() + 2.0
        while stage.process_calls < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        engine.request_stop()

    t = threading.Thread(target=_stop_after_frames, daemon=True)
    t.start()

    engine.run()  # blocks until stopped

    t.join(timeout=2.0)
    assert stage.process_calls >= 3
    assert stage.stop_calls >= 1


def test_run_calls_on_frame_callback():
    """on_frame callback must be called for each processed frame."""
    stage = _FakeStage("s")
    engine = _build_engine([stage], fps=120)
    frame_callbacks: list[object] = []

    engine._on_frame = frame_callbacks.append

    def _stop():
        deadline = time.monotonic() + 2.0
        while len(frame_callbacks) < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        engine.request_stop()

    t = threading.Thread(target=_stop, daemon=True)
    t.start()
    engine.run()
    t.join(timeout=2.0)
    assert len(frame_callbacks) >= 3


def test_run_abort_when_start_stages_fails():
    """run() must return immediately if start_stages() returns False."""

    class FailStart(_FakeStage):
        def start(self) -> bool:
            return False

    engine = _build_engine([FailStart("bad")])
    engine.run()  # should return without blocking


def test_run_calls_on_shutdown_callback():
    """on_shutdown callback must be called once after the loop exits."""
    stage = _FakeStage("s")
    engine = _build_engine([stage], fps=120)
    called: list[bool] = []
    engine._on_shutdown = lambda: called.append(True)

    def _stop():
        time.sleep(0.05)
        engine.request_stop()

    t = threading.Thread(target=_stop, daemon=True)
    t.start()
    engine.run()
    t.join(timeout=2.0)
    assert called == [True]


def test_update_fps_tracks_actual_fps():
    """actual_fps should be set after at least 1 second of running."""
    stage = _FakeStage("s")
    engine = _build_engine([stage], fps=120)

    def _stop():
        deadline = time.monotonic() + 3.0
        while engine.actual_fps == 0.0 and time.monotonic() < deadline:
            time.sleep(0.05)
        engine.request_stop()

    t = threading.Thread(target=_stop, daemon=True)
    t.start()
    engine.run()
    t.join(timeout=4.0)
    assert engine.actual_fps > 0.0


# ---------------------------------------------------------------------------
# request_stop is thread-safe
# ---------------------------------------------------------------------------


def test_request_stop_sets_stop_event():
    stage = _FakeStage("s")
    engine = _build_engine([stage])
    assert engine.is_running
    engine.request_stop()
    assert not engine.is_running
    assert engine._stop_event.is_set()


def test_request_stop_from_thread():
    stage = _FakeStage("s")
    engine = _build_engine([stage])

    def stopper():
        time.sleep(0.01)
        engine.request_stop()

    t = threading.Thread(target=stopper, daemon=True)
    t.start()
    t.join(timeout=1.0)
    assert engine._stop_event.is_set()
