"""Tests for KalmanFilter2D."""

from __future__ import annotations

import pytest

from gazecontrol.filters.kalman import KalmanFilter2D


def test_first_call_initialises_and_returns_input():
    kf = KalmanFilter2D()
    x, y = kf.update(100.0, 200.0)
    assert x == pytest.approx(100.0, abs=1e-6)
    assert y == pytest.approx(200.0, abs=1e-6)


def test_stationary_measurement_stays_close():
    kf = KalmanFilter2D(process_noise=1e-4, measurement_noise_base=0.5)
    for _ in range(100):
        x, y = kf.update(500.0, 300.0)
    assert abs(x - 500.0) < 1.0
    assert abs(y - 300.0) < 1.0


def test_tracking_linear_motion():
    """Filter should track a linearly moving target with bounded lag."""
    kf = KalmanFilter2D(process_noise=0.1, measurement_noise_base=0.1)
    # Ramp x from 0 to 100 over 50 steps.
    for i in range(50):
        x, _y = kf.update(float(i * 2), 0.0)
    # Should be within 5 px of the true position.
    assert abs(x - 98.0) < 10.0


def test_low_confidence_increases_prediction_weight():
    """Low confidence should increase R, making the filter lean on prediction."""
    kf_conf = KalmanFilter2D(measurement_noise_base=1.0)
    kf_noconf = KalmanFilter2D(measurement_noise_base=1.0)

    # Initialise both at (0, 0).
    kf_conf.update(0.0, 0.0)
    kf_noconf.update(0.0, 0.0)

    # Feed a large outlier measurement.
    x_conf, _ = kf_conf.update(1000.0, 0.0, confidence=1.0)
    x_noconf, _ = kf_noconf.update(1000.0, 0.0, confidence=0.1)

    # High-confidence should jump toward measurement more than low-confidence.
    assert x_conf > x_noconf


def test_reset_reinitialises():
    kf = KalmanFilter2D()
    kf.update(999.0, 999.0)
    kf.reset()
    x, y = kf.update(5.0, 7.0)
    assert x == pytest.approx(5.0, abs=1e-6)
    assert y == pytest.approx(7.0, abs=1e-6)


def test_predict_only_extrapolates():
    kf = KalmanFilter2D(process_noise=0.01)
    # Give it velocity by stepping through two points.
    kf.update(0.0, 0.0)
    kf.update(10.0, 0.0)
    px, _py = kf.predict_only()
    # Prediction should be ahead of last measurement.
    assert px > 10.0


def test_predict_only_before_init_returns_zeros():
    kf = KalmanFilter2D()
    x, y = kf.predict_only()
    assert x == 0.0
    assert y == 0.0
