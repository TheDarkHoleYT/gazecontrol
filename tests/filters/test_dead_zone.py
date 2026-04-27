"""Tests for DeadZone."""

from __future__ import annotations

import pytest

from gazecontrol.filters.dead_zone import DeadZone


def test_first_call_initialises_anchor():
    dz = DeadZone(radius_px=5, hysteresis_px=10)
    x, y = dz.apply(100.0, 200.0)
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(200.0)


def test_small_movement_within_radius_locked():
    """Movement inside radius_px should return the anchor (locked position)."""
    dz = DeadZone(radius_px=10, hysteresis_px=20)
    dz.apply(100.0, 100.0)  # initialise anchor
    # Move 5 px — within radius, should stay at anchor.
    x, y = dz.apply(105.0, 100.0)
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(100.0)


def test_large_movement_unlocks():
    """Movement exceeding hysteresis_px should unlock the filter."""
    dz = DeadZone(radius_px=5, hysteresis_px=10)
    dz.apply(0.0, 0.0)  # initialise anchor at (0, 0)
    dz.apply(3.0, 0.0)  # still locked (within radius)
    x, y = dz.apply(20.0, 0.0)  # exceeds hysteresis → unlock
    assert x == pytest.approx(20.0)
    assert y == pytest.approx(0.0)


def test_relocks_when_settled():
    """After unlocking, two consecutive near-identical positions should relock.

    In unlocked state the anchor follows each position, so relocking happens
    when the *next* position is within radius_px of the *previous* position
    (i.e. the pointer has effectively stopped moving).
    """
    dz = DeadZone(radius_px=5, hysteresis_px=15)
    dz.apply(0.0, 0.0)  # initialise anchor at (0, 0)
    dz.apply(30.0, 0.0)  # unlock; anchor moves to (30, 0)
    dz.apply(31.0, 0.0)  # still unlocked; anchor moves to (31, 0); dist=1<5 → lock!
    assert dz._locked is True


def test_reset_clears_anchor():
    dz = DeadZone(radius_px=5, hysteresis_px=10)
    dz.apply(500.0, 500.0)
    dz.reset()
    assert dz._anchor_x is None
    assert dz._anchor_y is None
    assert dz._locked is False


def test_hysteresis_clamped_to_radius():
    """hysteresis_px < radius_px is silently raised to radius_px."""
    dz = DeadZone(radius_px=10, hysteresis_px=2)
    assert dz._r_out >= dz._r_in


def test_no_jitter_under_dead_zone():
    """Rapid sub-radius oscillation should produce constant output."""
    dz = DeadZone(radius_px=8, hysteresis_px=16)
    dz.apply(0.0, 0.0)  # set anchor

    results = set()
    for i in range(20):
        # Oscillate ±4 px (within radius).
        x, y = dz.apply(4.0 * ((-1) ** i), 0.0)
        results.add((round(x, 1), round(y, 1)))

    # Should always return the same anchor position.
    assert len(results) == 1
