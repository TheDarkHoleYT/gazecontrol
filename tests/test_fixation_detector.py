"""Test per FixationDetector (I-VT)."""
from gazecontrol.gaze.fixation_detector import FixationDetector, GazeEvent


def _make_detector():
    # px_per_deg=44 → fixation_thr=30*44=1320 px/s, saccade_thr=100*44=4400 px/s
    return FixationDetector(screen_px_per_degree=44.0)


def test_still_gaze_is_fixation():
    """Un punto fermo deve essere classificato come FIXATION."""
    det = _make_detector()
    evt = None
    for i in range(20):
        evt = det.update(500.0, 400.0, timestamp=i / 30.0)
    assert evt.type == FixationDetector.FIXATION


def test_fast_jump_is_saccade():
    """Un salto di 500px in 1 frame a 30fps = 15000 px/s → SACCADE."""
    det = _make_detector()
    det.update(0.0, 0.0, timestamp=0.0)
    evt = det.update(500.0, 0.0, timestamp=1 / 30.0)
    assert evt.type == FixationDetector.SACCADE


def test_blink_returns_blink_type():
    det = _make_detector()
    evt = det.update(500.0, 400.0, timestamp=0.0, is_blink=True)
    assert evt.type == FixationDetector.BLINK


def test_fixation_centroid_computed():
    """Dopo abbastanza campioni in fixation, il centroide deve essere calcolato."""
    det = FixationDetector(screen_px_per_degree=44.0, fixation_history_ms=150.0)
    evt = None
    # Genera campioni fissi per > 150ms
    for i in range(10):
        evt = det.update(300.0, 200.0, timestamp=i * 0.02)
    assert evt.centroid is not None
    cx, cy = evt.centroid
    assert abs(cx - 300.0) < 5.0
    assert abs(cy - 200.0) < 5.0


def test_saccade_clears_fixation_buffer():
    """Dopo una saccade, il buffer fixation deve essere vuoto (centroid=None)."""
    det = _make_detector()
    for i in range(10):
        det.update(300.0, 200.0, timestamp=i / 30.0)
    # Saccade
    evt = det.update(900.0, 600.0, timestamp=10 / 30.0)
    assert evt.type == FixationDetector.SACCADE
    assert evt.centroid is None


def test_reset():
    det = _make_detector()
    det.update(500.0, 400.0, timestamp=0.0)
    det.reset()
    # Dopo reset nessun prev_point → velocity = 0 → fixation
    evt = det.update(500.0, 400.0, timestamp=0.1)
    assert evt.velocity_px_s == 0.0
    assert evt.type == FixationDetector.FIXATION


if __name__ == '__main__':
    test_still_gaze_is_fixation()
    test_fast_jump_is_saccade()
    test_blink_returns_blink_type()
    test_fixation_centroid_computed()
    test_saccade_clears_fixation_buffer()
    test_reset()
    print("Tutti i test FixationDetector superati.")
