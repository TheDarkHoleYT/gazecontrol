"""I-VT fixation detector classification tests."""

from __future__ import annotations

from gazecontrol.gaze.fixation_detector import FixationDetector


def test_initial_sample_is_fixation():
    fd = FixationDetector()
    ev = fd.update(100.0, 100.0, timestamp=0.0)
    assert ev.type == "fixation"
    assert ev.velocity_px_s == 0.0


def test_fast_motion_classifies_as_saccade():
    fd = FixationDetector(fixation_vel_thr=30.0, saccade_vel_thr=100.0, screen_px_per_degree=44.0)
    fd.update(0.0, 0.0, timestamp=0.0)
    # 5000 px in 0.01 s = 500_000 px/s → way above saccade threshold
    ev = fd.update(5000.0, 0.0, timestamp=0.01)
    assert ev.type == "saccade"


def test_slow_motion_classifies_as_fixation():
    fd = FixationDetector(fixation_vel_thr=30.0, saccade_vel_thr=100.0, screen_px_per_degree=44.0)
    fd.update(0.0, 0.0, timestamp=0.0)
    # 10 px in 0.1 s = 100 px/s, < fixation threshold (30 * 44 = 1320)
    ev = fd.update(10.0, 0.0, timestamp=0.1)
    assert ev.type == "fixation"


def test_blink_event():
    fd = FixationDetector()
    fd.update(0.0, 0.0, timestamp=0.0)
    ev = fd.update(0.0, 0.0, timestamp=0.05, is_blink=True)
    assert ev.type == "blink"


def test_fixation_centroid_after_multiple_samples():
    fd = FixationDetector()
    for i in range(10):
        ev = fd.update(100.0 + i * 0.1, 200.0, timestamp=i * 0.01)
    assert ev.centroid is not None
    assert abs(ev.centroid[0] - 100.45) < 1.0
    assert ev.centroid[1] == 200.0


def test_reset_clears_history():
    fd = FixationDetector()
    fd.update(100.0, 200.0, timestamp=0.0)
    fd.reset()
    ev = fd.update(500.0, 600.0, timestamp=0.01)
    assert ev.velocity_px_s == 0.0
