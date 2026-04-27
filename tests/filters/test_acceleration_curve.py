"""Tests for AccelerationCurve."""

from __future__ import annotations

import pytest

from gazecontrol.filters.acceleration_curve import AccelerationCurve


def test_slow_velocity_returns_gain_low():
    curve = AccelerationCurve(v_low=0.05, v_high=1.2, gain_low=0.7, gain_high=1.8)
    assert curve.gain(0.0) == pytest.approx(0.7)
    assert curve.gain(0.04) == pytest.approx(0.7)


def test_fast_velocity_returns_gain_high():
    curve = AccelerationCurve(v_low=0.05, v_high=1.2, gain_low=0.7, gain_high=1.8)
    assert curve.gain(1.2) == pytest.approx(1.8)
    assert curve.gain(5.0) == pytest.approx(1.8)


def test_mid_velocity_is_between_gains():
    curve = AccelerationCurve(v_low=0.0, v_high=1.0, gain_low=0.5, gain_high=2.0)
    g = curve.gain(0.5)
    assert 0.5 < g < 2.0


def test_gain_monotonically_increasing():
    """Higher velocity → higher gain."""
    curve = AccelerationCurve(v_low=0.1, v_high=1.0, gain_low=0.5, gain_high=2.0)
    velocities = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
    gains = [curve.gain(v) for v in velocities]
    for i in range(len(gains) - 1):
        assert gains[i] <= gains[i + 1]


def test_apply_scales_displacement():
    curve = AccelerationCurve(v_low=0.0, v_high=1.0, gain_low=1.0, gain_high=2.0)
    # velocity 0 → gain_low=1.0 → dx unchanged
    dx, dy = curve.apply(0.0, 0.0)
    assert dx == pytest.approx(0.0)
    assert dy == pytest.approx(0.0)


def test_apply_direction_preserved():
    """The direction of the displacement should be preserved."""
    curve = AccelerationCurve(v_low=0.0, v_high=2.0, gain_low=0.5, gain_high=1.5)
    dx, dy = curve.apply(-3.0, 4.0)
    assert dx < 0
    assert dy > 0


def test_curve_smooth_no_discontinuity():
    """Gain should not jump abruptly at v_low / v_high boundaries."""
    curve = AccelerationCurve(v_low=0.5, v_high=1.0, gain_low=0.8, gain_high=1.5)
    eps = 1e-5
    g_below = curve.gain(0.5 - eps)
    g_at = curve.gain(0.5)
    g_above = curve.gain(0.5 + eps)
    # Transition should be smooth (no jump > 0.1).
    assert abs(g_at - g_below) < 0.1
    assert abs(g_above - g_at) < 0.1
