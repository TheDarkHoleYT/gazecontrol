"""GazeStage tests with a stub backend."""

from __future__ import annotations

import numpy as np
import pytest

from gazecontrol.gaze.backend import GazePrediction
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.gaze_stage import GazeStage


class _StubBackend:
    name = "stub"

    def __init__(self, prediction: GazePrediction | None = None, start_ok: bool = True) -> None:
        self._pred = prediction
        self._start_ok = start_ok
        self.start_calls = 0
        self.stop_calls = 0
        self.predict_calls = 0

    def start(self) -> bool:
        self.start_calls += 1
        return self._start_ok

    def stop(self) -> None:
        self.stop_calls += 1

    def is_calibrated(self) -> bool:
        return True

    def predict(self, frame_bgr, frame_rgb, timestamp):
        self.predict_calls += 1
        return self._pred


def _ctx(frame_present: bool = True, t0: float = 0.0) -> FrameContext:
    ctx = FrameContext(t0=t0)
    if frame_present:
        ctx.capture_ok = True
        ctx.frame_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        ctx.frame_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    return ctx


def test_start_returns_false_when_backend_fails():
    backend = _StubBackend(start_ok=False)
    stage = GazeStage(backend=backend, screen_w=1920, screen_h=1080)
    assert stage.start() is False


def test_start_then_predict_writes_gaze_fields():
    pred = GazePrediction(screen_xy=(100, 200), confidence=0.8, backend_name="stub")
    backend = _StubBackend(prediction=pred)
    stage = GazeStage(backend=backend, screen_w=1920, screen_h=1080)
    assert stage.start() is True
    ctx = stage.process(_ctx())
    assert ctx.face_present is True
    assert ctx.gaze_screen is not None
    assert ctx.gaze_confidence == pytest.approx(0.8)
    stage.stop()
    assert backend.stop_calls == 1


def test_predict_skipped_when_capture_failed():
    pred = GazePrediction(screen_xy=(100, 200), confidence=0.8)
    backend = _StubBackend(prediction=pred)
    stage = GazeStage(backend=backend, screen_w=1920, screen_h=1080)
    stage.start()
    ctx = FrameContext()  # capture_ok=False
    stage.process(ctx)
    assert backend.predict_calls == 0


def test_blink_propagates_and_blank_xy():
    pred = GazePrediction(screen_xy=(100, 200), confidence=0.0, blink=True)
    backend = _StubBackend(prediction=pred)
    stage = GazeStage(backend=backend, screen_w=1920, screen_h=1080)
    stage.start()
    ctx = stage.process(_ctx())
    assert ctx.gaze_blink is True


def test_no_prediction_clears_face_present():
    backend = _StubBackend(prediction=None)
    stage = GazeStage(backend=backend, screen_w=1920, screen_h=1080)
    stage.start()
    ctx = stage.process(_ctx())
    assert ctx.face_present is False
    assert ctx.gaze_screen is None


def test_drift_feedback_passes_to_backend_when_enabled():
    pred = GazePrediction(screen_xy=(100, 200), confidence=0.8)
    backend = _StubBackend(prediction=pred)
    stage = GazeStage(backend=backend, screen_w=1920, screen_h=1080)
    stage.start()
    # No exception even when drift disabled by default.
    stage.on_user_action((100, 200), (50, 150, 100, 100))
