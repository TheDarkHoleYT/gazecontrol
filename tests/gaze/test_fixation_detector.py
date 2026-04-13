"""Extended tests for FixationDetector (I-VT)."""
from __future__ import annotations

import time

from gazecontrol.gaze.fixation_detector import FixationDetector, GazeEvent


def test_initial_event_type():
    fd = FixationDetector()
    t = time.monotonic()
    ev = fd.update(500.0, 300.0, t)
    assert isinstance(ev, GazeEvent)
    assert ev.type in ("fixation", "saccade", "pursuit", "blink")


def test_stationary_gaze_becomes_fixation():
    fd = FixationDetector()
    t = time.monotonic()
    last = None
    # Feed 50 identical points at 30fps → should settle to fixation.
    for i in range(50):
        last = fd.update(500.0, 300.0, t + i / 30.0)
    assert last is not None
    assert last.type == "fixation"


def test_fast_movement_becomes_saccade():
    fd = FixationDetector()
    t = time.monotonic()
    # Jump 2000 px in 1/30 s → very fast.
    fd.update(0.0, 0.0, t)
    ev = fd.update(2000.0, 0.0, t + 1 / 30.0)
    assert ev.type == "saccade"


def test_fixation_centroid_computed():
    fd = FixationDetector()
    t = time.monotonic()
    for i in range(50):
        ev = fd.update(500.0, 300.0, t + i / 30.0)
    # After settling, centroid should be near the fixated point.
    assert ev.centroid is not None
    cx, cy = ev.centroid
    assert abs(cx - 500.0) < 20.0
    assert abs(cy - 300.0) < 20.0


def test_event_has_velocity():
    fd = FixationDetector()
    t = time.monotonic()
    fd.update(0.0, 0.0, t)
    ev = fd.update(100.0, 0.0, t + 1 / 30.0)
    assert ev.velocity_px_s >= 0.0
