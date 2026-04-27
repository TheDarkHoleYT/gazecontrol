"""Verify GazeBackend Protocol structural typing + GazePrediction value object."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from gazecontrol.gaze.backend import GazeBackend, GazePrediction


class _StubBackend:
    name = "stub"

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> bool:
        self.started = True
        return True

    def stop(self) -> None:
        self.stopped = True

    def is_calibrated(self) -> bool:
        return True

    def predict(self, frame_bgr, frame_rgb, timestamp):
        return GazePrediction(
            screen_xy=(100, 200),
            confidence=0.8,
            backend_name=self.name,
        )


def test_stub_satisfies_protocol():
    backend = _StubBackend()
    assert isinstance(backend, GazeBackend)


def test_gaze_prediction_is_frozen():
    p = GazePrediction(screen_xy=(1, 2), confidence=0.5)
    with pytest.raises(FrozenInstanceError):
        p.confidence = 0.9  # type: ignore[misc]


def test_stub_backend_full_lifecycle():
    backend = _StubBackend()
    assert backend.start() is True
    assert backend.is_calibrated() is True
    pred = backend.predict(
        np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1, 1, 3), dtype=np.uint8), 0.0
    )
    assert pred is not None
    assert pred.screen_xy == (100, 200)
    backend.stop()
    assert backend.stopped is True
