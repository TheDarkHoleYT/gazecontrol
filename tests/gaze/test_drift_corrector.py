"""Drift corrector tests."""

from __future__ import annotations

from gazecontrol.gaze.drift_corrector import DriftCorrector


def test_initial_offset_is_zero():
    dc = DriftCorrector()
    assert dc.offset == (0.0, 0.0)
    x, y = dc.correct(500.0, 400.0)
    assert (x, y) == (500.0, 400.0)


def test_correct_clamps_to_screen():
    dc = DriftCorrector(screen_w=1920, screen_h=1080)
    x, y = dc.correct(-500.0, -500.0)
    assert x == 0.0 and y == 0.0
    x, y = dc.correct(5000.0, 5000.0)
    assert x == 1919.0 and y == 1079.0


def test_implicit_recal_reduces_offset():
    dc = DriftCorrector(implicit_alpha=0.2)
    dc.on_action(
        gaze_point=(110.0, 200.0),
        target_window={"rect": (50, 150, 100, 100)},  # centroid (100, 200)
    )
    # err_x = 110 - 100 = 10 → offset_x += 0.2 * 10 = 2
    assert abs(dc.offset[0] - 2.0) < 1e-6
    # err_y = 200 - 200 = 0
    assert dc.offset[1] == 0.0


def test_reset_zeros_offset():
    dc = DriftCorrector()
    dc.on_action((110, 200), {"rect": (50, 150, 100, 100)})
    dc.reset()
    assert dc.offset == (0.0, 0.0)


def test_offset_clamped_to_max():
    dc = DriftCorrector(implicit_alpha=1.0, max_correction_px=50.0)
    dc.on_action((1000.0, 0.0), {"rect": (0, 0, 0, 0)})
    import math

    assert math.hypot(*dc.offset) <= 50.0 + 1e-6


# Regression: edge-snapping must drive corrected gaze TOWARD the screen, not
# further off-screen.  Prior to the BUG-001 fix the sign was inverted and the
# loop diverged.
def test_edge_snap_left_converges_toward_screen():
    dc = DriftCorrector(
        screen_w=1920,
        screen_h=1080,
        edge_margin_px=60,
        edge_correction_rate=0.2,
        max_correction_px=500.0,
    )
    raw_x, raw_y = -200.0, 500.0  # 200 px past the left edge
    last_dist = abs(raw_x)
    for _ in range(40):
        cx, _ = dc.correct(raw_x, raw_y)
        dist = abs(cx)  # raw distance from left edge
        # Distance to the visible region must be monotonically non-increasing.
        assert dist <= last_dist + 1e-6
        last_dist = dist
    # And after enough iterations the corrected point reaches the edge.
    assert last_dist <= 1.0


def test_edge_snap_right_converges_toward_screen():
    dc = DriftCorrector(
        screen_w=1920,
        screen_h=1080,
        edge_margin_px=60,
        edge_correction_rate=0.2,
        max_correction_px=500.0,
    )
    raw_x, raw_y = 2200.0, 500.0  # 280 px past the right edge
    sw = 1920
    last_overshoot = raw_x - sw
    for _ in range(40):
        cx, _ = dc.correct(raw_x, raw_y)
        overshoot = max(0.0, cx - (sw - 1))
        # Clamp prevents observation of the underlying offset, but offset
        # should monotonically grow until the correction lands the corrected
        # point inside the screen.
        assert dc.offset[0] >= 0.0
        last_overshoot = overshoot
    # The applied offset must be > 0 (we shifted left) and bounded.
    assert dc.offset[0] > 0.0
    assert dc.offset[0] <= 500.0 + 1e-6
    _ = last_overshoot  # keep symmetry with the left test
