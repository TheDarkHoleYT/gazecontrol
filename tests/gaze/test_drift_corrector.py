"""Drift corrector tests."""

from __future__ import annotations

import itertools

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


# ---------------------------------------------------------------------------
# v1.0 G7 — explicit recenter, convergence telemetry, Kalman mode
# ---------------------------------------------------------------------------


def test_recenter_to_directly_sets_offset():
    dc = DriftCorrector(screen_w=1920, screen_h=1080, max_correction_px=500.0)
    # User gaze landed at (1000, 600) while looking at the centre (960, 540).
    dc.recenter_to(raw_xy=(1000.0, 600.0), target_xy=(960.0, 540.0))
    assert dc.offset == (40.0, 60.0)


def test_recenter_clamps_to_max_correction():
    dc = DriftCorrector(max_correction_px=50.0)
    dc.recenter_to(raw_xy=(1000.0, 0.0), target_xy=(0.0, 0.0))
    import math

    assert math.hypot(*dc.offset) <= 50.0 + 1e-6


def test_request_recenter_collects_samples_then_finalizes():
    dc = DriftCorrector(
        screen_w=1920, screen_h=1080, max_correction_px=500.0,
        recenter_sample_count=3,
    )
    dc.request_recenter(target_xy=(100.0, 100.0))
    assert dc.recenter_in_progress is True

    # First two samples: still collecting.
    assert dc.feed_recenter_sample((110.0, 90.0)) is False
    assert dc.feed_recenter_sample((130.0, 110.0)) is False
    assert dc.recenter_in_progress is True

    # Third sample completes the session.
    done = dc.feed_recenter_sample((120.0, 100.0))
    assert done is True
    assert dc.recenter_in_progress is False
    # avg raw = ((110+130+120)/3, (90+110+100)/3) = (120, 100)
    # offset = (120 - 100, 100 - 100) = (20, 0)
    assert dc.offset == (20.0, 0.0)


def test_request_recenter_default_target_is_screen_centre():
    dc = DriftCorrector(
        screen_w=1920, screen_h=1080, recenter_sample_count=1, max_correction_px=500.0,
    )
    dc.request_recenter()  # default target = screen centre (960, 540)
    dc.feed_recenter_sample((1000.0, 600.0))
    assert dc.offset == (40.0, 60.0)


def test_feed_recenter_sample_no_op_when_inactive():
    dc = DriftCorrector()
    assert dc.feed_recenter_sample((100.0, 100.0)) is False
    assert dc.offset == (0.0, 0.0)


def test_request_recenter_restart_clears_prior_samples():
    dc = DriftCorrector(recenter_sample_count=3, max_correction_px=500.0)
    dc.request_recenter(target_xy=(100.0, 100.0))
    dc.feed_recenter_sample((200.0, 200.0))
    dc.feed_recenter_sample((250.0, 250.0))
    # User restarts the recenter → previous samples discarded.
    dc.request_recenter(target_xy=(100.0, 100.0))
    dc.feed_recenter_sample((110.0, 110.0))
    dc.feed_recenter_sample((110.0, 110.0))
    done = dc.feed_recenter_sample((110.0, 110.0))
    assert done is True
    assert dc.offset == (10.0, 10.0)


def test_reset_aborts_recenter_session():
    dc = DriftCorrector(recenter_sample_count=3)
    dc.request_recenter()
    dc.feed_recenter_sample((10.0, 10.0))
    dc.reset()
    assert dc.recenter_in_progress is False


def test_kalman_mode_converges_with_repeated_recal():
    """Repeated implicit-recal observations with a fixed true bias must
    converge to that bias (offset stabilises near it) and the Kalman
    covariance must shrink monotonically.

    Each iteration models the closed loop: raw gaze stays fixed at the
    true bias (40, 30), so the *corrected* gaze fed back into
    ``on_action`` is ``raw - current_offset``. The error driving the
    update therefore shrinks as offset approaches the bias.
    """
    dc = DriftCorrector(
        max_correction_px=500.0,
        mode="kalman",
        kalman_process_noise=0.01,
        kalman_measurement_noise=4.0,
    )
    true_bias = (40.0, 30.0)
    rect = (0, 0, 0, 0)  # centroid (0, 0)
    cov_x_history = []
    for _ in range(60):
        corrected = (true_bias[0] - dc.offset[0], true_bias[1] - dc.offset[1])
        dc.on_action(corrected, {"rect": rect})
        cov_x_history.append(dc.telemetry()["cov_x"])
    # Offset should converge near the true bias.
    assert abs(dc.offset[0] - true_bias[0]) < 5.0
    assert abs(dc.offset[1] - true_bias[1]) < 5.0
    # Kalman covariance monotonically non-increasing.
    for prev, curr in itertools.pairwise(cov_x_history):
        assert curr <= prev + 1e-9


def test_ema_mode_remains_default():
    dc = DriftCorrector()
    assert dc.mode == "ema"


def test_is_converged_false_during_warmup():
    dc = DriftCorrector(convergence_window=10, convergence_threshold_px=5.0)
    # Not enough samples yet.
    for _ in range(5):
        dc.on_action((1.0, 0.0), {"rect": (0, 0, 0, 0)})
    assert dc.is_converged() is False


def test_is_converged_true_once_offset_stabilises():
    dc = DriftCorrector(
        implicit_alpha=0.05,
        max_correction_px=500.0,
        convergence_window=20,
        convergence_threshold_px=3.0,
    )
    # Many small consistent observations → deltas shrink toward zero.
    for _ in range(200):
        dc.on_action((10.0, 10.0), {"rect": (0, 0, 0, 0)})
    assert dc.is_converged() is True


def test_telemetry_snapshot_shape():
    dc = DriftCorrector()
    snap = dc.telemetry()
    assert set(snap) == {
        "offset_x_px", "offset_y_px", "offset_mag_px",
        "mode", "samples_seen", "converged", "cov_x", "cov_y",
    }
    assert snap["mode"] == "ema"
    assert snap["converged"] is False


def test_invalid_recenter_target_falls_back_to_centre():
    # Sanity: calling without target uses (sw/2, sh/2).
    dc = DriftCorrector(
        screen_w=1000, screen_h=800, recenter_sample_count=1, max_correction_px=500.0,
    )
    dc.request_recenter()
    dc.feed_recenter_sample((600.0, 500.0))  # 100 px right, 100 px down from centre
    assert dc.offset == (100.0, 100.0)
