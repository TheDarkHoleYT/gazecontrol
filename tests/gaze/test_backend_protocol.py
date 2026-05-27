"""Verify GazeBackend Protocol structural typing + GazePrediction value object."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from gazecontrol.gaze.backend import GazeBackend, GazePrediction, GazeQuality


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


# ---------------------------------------------------------------------------
# v1.0 GazePrediction extensions (uncertainty, head pose, face bbox, quality)
# ---------------------------------------------------------------------------


def test_gaze_prediction_v10_fields_default_to_none_or_zero():
    """The new v1.0 fields (uncertainty, head_pose, face_bbox, face_id,
    quality_flags) must default such that legacy code that only passes
    screen_xy + confidence continues to work unchanged."""
    p = GazePrediction(screen_xy=(10, 20), confidence=0.5)
    assert p.uncertainty_px is None
    assert p.head_pose_rad is None
    assert p.face_bbox_norm is None
    assert p.face_id is None
    assert p.quality_flags == 0


def test_gaze_prediction_carries_uncertainty_and_head_pose():
    p = GazePrediction(
        screen_xy=(10, 20),
        confidence=0.7,
        uncertainty_px=42.0,
        head_pose_rad=(0.1, -0.2, 0.05),
        face_bbox_norm=(0.1, 0.1, 0.9, 0.9),
        face_id=7,
        quality_flags=int(GazeQuality.OFF_AXIS | GazeQuality.MULTI_FACE),
    )
    assert p.uncertainty_px == 42.0
    assert p.head_pose_rad == (0.1, -0.2, 0.05)
    assert p.face_bbox_norm == (0.1, 0.1, 0.9, 0.9)
    assert p.face_id == 7
    assert p.quality_flags & GazeQuality.OFF_AXIS
    assert p.quality_flags & GazeQuality.MULTI_FACE
    assert not p.quality_flags & GazeQuality.BLINK


def test_quality_flags_are_distinct_powers_of_two():
    """Each flag must occupy a unique bit so ORs compose correctly."""
    flags = [
        GazeQuality.BLINK,
        GazeQuality.LOW_LIGHT,
        GazeQuality.OFF_AXIS,
        GazeQuality.OCCLUDED,
        GazeQuality.MULTI_FACE,
    ]
    seen = 0
    for f in flags:
        assert int(f) > 0
        assert int(f) & (int(f) - 1) == 0  # power of two
        assert int(f) & seen == 0  # disjoint from earlier flags
        seen |= int(f)
    assert GazeQuality.NONE == 0


def test_gaze_prediction_still_frozen_with_v10_fields():
    p = GazePrediction(
        screen_xy=(1, 2),
        confidence=0.5,
        uncertainty_px=10.0,
        quality_flags=int(GazeQuality.BLINK),
    )
    with pytest.raises(FrozenInstanceError):
        p.uncertainty_px = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        p.quality_flags = 0  # type: ignore[misc]
