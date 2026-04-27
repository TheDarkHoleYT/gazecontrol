"""Tests for OneEuroFilter."""

from __future__ import annotations

import math

import pytest

from gazecontrol.filters.one_euro import OneEuroFilter


def test_first_call_returns_input():
    f = OneEuroFilter()
    assert f.filter(5.0, timestamp=0.0) == pytest.approx(5.0, abs=1e-9)


def test_stationary_signal_converges():
    """A constant signal should converge to itself after many samples."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    ts = 0.0
    for _ in range(200):
        ts += 1.0 / 30.0
        v = f.filter(100.0, timestamp=ts)
    assert abs(v - 100.0) < 0.01


def test_jitter_reduced_vs_raw():
    """Filter output jitter should be lower than raw jitter for a noisy signal."""
    import random

    rng = random.Random(42)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    raw_vals, filt_vals = [], []
    ts = 0.0
    for _ in range(150):
        ts += 1.0 / 30.0
        raw = 500.0 + rng.gauss(0, 5)
        raw_vals.append(raw)
        filt_vals.append(f.filter(raw, timestamp=ts))

    raw_std = (sum((v - 500) ** 2 for v in raw_vals) / len(raw_vals)) ** 0.5
    filt_std = (sum((v - 500) ** 2 for v in filt_vals) / len(filt_vals)) ** 0.5
    assert filt_std < raw_std


def test_reset_clears_state():
    f = OneEuroFilter()
    f.filter(999.0, timestamp=0.0)
    f.reset()
    # After reset, first call should initialise to the new value.
    v = f.filter(1.0, timestamp=0.0)
    assert v == pytest.approx(1.0, abs=1e-9)


def test_high_beta_responsive_to_fast_motion():
    """With high beta the filter should follow fast ramps closely."""
    f_slow = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f_fast = OneEuroFilter(min_cutoff=1.0, beta=1.0)
    ts = 0.0
    for i in range(60):
        ts += 1.0 / 30.0
        val = float(i * 10)
        f_slow.filter(val, timestamp=ts)
        f_fast.filter(val, timestamp=ts)

    # At the end, fast-beta filter should be closer to the ramp.
    v_slow = f_slow.filter(590.0, timestamp=ts + 1 / 30)
    v_fast = f_fast.filter(590.0, timestamp=ts + 1 / 30)
    assert v_fast > v_slow


def test_none_timestamp_uses_monotonic(monkeypatch):
    """Passing timestamp=None should use time.monotonic() without crashing."""
    f = OneEuroFilter()
    v = f.filter(42.0)
    assert math.isfinite(v)


def test_frequency_adapts_from_timestamps():
    """Filter frequency estimate should update from actual inter-frame dt."""
    f = OneEuroFilter(freq=30.0)
    # Feed at 10 Hz.
    ts = 0.0
    for _ in range(10):
        ts += 0.1
        f.filter(1.0, timestamp=ts)
    # freq should now be close to 10.
    assert abs(f._freq - 10.0) < 2.0
