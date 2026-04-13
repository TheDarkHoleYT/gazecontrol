"""Extended tests for OneEuroFilter — adaptive smoothing."""
from __future__ import annotations

from gazecontrol.gaze.one_euro_filter import OneEuroFilter


def test_filter_reduces_noise():
    """Filter output should be smoother than raw noisy signal."""
    import numpy as np

    rng = np.random.default_rng(0)
    f = OneEuroFilter(freq=30, min_cutoff=1.5, beta=0.007)

    # Constant signal + Gaussian noise.
    t = 0.0
    raw = []
    filtered = []
    for _ in range(100):
        noisy = 500.0 + rng.normal(0, 10)
        raw.append(noisy)
        filtered.append(f.filter(noisy, timestamp=t))
        t += 1 / 30

    raw_std = float(np.std(raw))
    filt_std = float(np.std(filtered[20:]))  # skip warmup
    assert filt_std < raw_std * 0.5, "Filter should reduce noise variance by >50%"


def test_filter_reacts_to_fast_changes():
    """During saccade (fast change), beta should increase cutoff → less lag."""
    f = OneEuroFilter(freq=30, min_cutoff=1.5, beta=0.5)  # high beta

    # Warm up at position 0.
    t = 0.0
    for _ in range(10):
        f.filter(0.0, timestamp=t)
        t += 1 / 30

    # Sudden jump to 1000 px.
    out = []
    for _ in range(10):
        out.append(f.filter(1000.0, timestamp=t))
        t += 1 / 30

    # With high beta, should reach >500 within 10 frames.
    assert out[-1] > 500, "High-beta filter should track fast changes quickly"


def test_reset_clears_state():
    f = OneEuroFilter(freq=30, min_cutoff=1.5, beta=0.007)
    f.filter(500.0, timestamp=0.0)
    f.reset()
    # After reset, next value should pass through unfiltered (or near-unfiltered).
    out = f.filter(100.0, timestamp=0.033)
    assert abs(out - 100.0) < 50.0  # near 100 after reset


def test_filter_monotonic_timestamps():
    """Filter should not crash with monotonically increasing timestamps."""
    f = OneEuroFilter(freq=60, min_cutoff=1.0, beta=0.01)
    t = 0.0
    for i in range(200):
        f.filter(float(i % 100), timestamp=t)
        t += 1 / 60
