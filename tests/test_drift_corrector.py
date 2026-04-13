"""Test per DriftCorrector."""
from gazecontrol.gaze.drift_corrector import DriftCorrector


def test_no_drift_initially():
    dc = DriftCorrector(screen_w=1920, screen_h=1080)
    cx, cy = dc.correct(500.0, 400.0)
    assert abs(cx - 500.0) < 1.0
    assert abs(cy - 400.0) < 1.0


def test_clamps_to_screen_bounds():
    dc = DriftCorrector(screen_w=1920, screen_h=1080)
    cx, cy = dc.correct(-100.0, 2000.0)
    assert cx >= 0.0
    assert cy <= 1079.0


def test_implicit_recal_updates_offset():
    dc = DriftCorrector(screen_w=1920, screen_h=1080, implicit_alpha=1.0)
    # gaze punta a (600, 400) ma finestra è centrata in (500, 400)
    target = {'rect': (400, 300, 200, 200)}  # centro = (500, 400)
    dc.on_action((600.0, 400.0), target)
    ox, oy = dc.offset
    # Offset dovrebbe essere positivo (gaze era troppo a destra)
    assert ox > 0, f"offset_x atteso >0, ottenuto {ox}"


def test_reset_clears_offset():
    dc = DriftCorrector(screen_w=1920, screen_h=1080)
    target = {'rect': (400, 300, 200, 200)}
    dc.on_action((800.0, 600.0), target)
    dc.reset()
    assert dc.offset == (0.0, 0.0)


def test_edge_snapping_corrects_left_overflow():
    dc = DriftCorrector(screen_w=1920, screen_h=1080, edge_margin_px=50,
                        edge_correction_rate=0.5, max_correction_px=500)
    # Simula molti frame con gaze fuori dallo schermo a sinistra
    for _ in range(20):
        dc.correct(-200.0, 500.0)
    ox, _ = dc.offset
    # L'offset dovrebbe essere aumentato per correggere il drift verso sinistra
    assert ox > 0, f"Edge snapping non ha corretto: offset_x={ox}"


def test_max_correction_cap():
    dc = DriftCorrector(screen_w=1920, screen_h=1080, max_correction_px=100.0,
                        implicit_alpha=1.0)
    # Errore enorme
    target = {'rect': (100, 100, 200, 200)}  # centro = (200, 200)
    dc.on_action((2000.0, 2000.0), target)
    import math
    mag = math.hypot(*dc.offset)
    assert mag <= 100.0 + 1e-3, f"Correzione non cappata: {mag:.1f}px"


if __name__ == '__main__':
    test_no_drift_initially()
    test_clamps_to_screen_bounds()
    test_implicit_recal_updates_offset()
    test_reset_clears_offset()
    test_edge_snapping_corrects_left_overflow()
    test_max_correction_cap()
    print("Tutti i test DriftCorrector superati.")
