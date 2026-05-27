"""Tests for the per-frame gaze confidence helpers (G1)."""

from __future__ import annotations

import numpy as np
import pytest

from gazecontrol.gaze.confidence import (
    AngleJitter,
    confidence_score,
    laplacian_variance,
)

# ---------------------------------------------------------------------------
# laplacian_variance
# ---------------------------------------------------------------------------


def test_laplacian_variance_zero_for_constant_image():
    """A flat patch has zero Laplacian variance — perfectly defocused."""
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert laplacian_variance(flat) == pytest.approx(0.0, abs=1e-6)


def test_laplacian_variance_positive_for_sharp_pattern():
    """A high-frequency checkerboard should score significantly above zero."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    assert laplacian_variance(noise) > 100.0


def test_laplacian_variance_grayscale_input_supported():
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    assert laplacian_variance(gray) > 0.0


def test_laplacian_variance_returns_zero_for_empty_input():
    assert laplacian_variance(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


# ---------------------------------------------------------------------------
# AngleJitter
# ---------------------------------------------------------------------------


def test_angle_jitter_zero_before_two_samples():
    j = AngleJitter()
    assert j.score() == 0.0
    j.push(10.0, 5.0)
    assert j.score() == 0.0


def test_angle_jitter_low_for_steady_stream():
    j = AngleJitter(window=5, jitter_saturation_deg=8.0)
    for _ in range(10):
        j.push(10.0, 5.0)
    assert j.score() == pytest.approx(0.0, abs=1e-6)


def test_angle_jitter_high_for_alternating_samples():
    j = AngleJitter(window=4, jitter_saturation_deg=8.0)
    for i in range(8):
        j.push(-10.0 if i % 2 == 0 else 10.0, 5.0)
    assert j.score() > 0.8  # saturated at the configured ceiling


def test_angle_jitter_reset_clears_buffer():
    j = AngleJitter(window=4)
    for i in range(8):
        j.push(-10.0 if i % 2 == 0 else 10.0, 5.0)
    j.reset()
    assert j.score() == 0.0


# ---------------------------------------------------------------------------
# confidence_score
# ---------------------------------------------------------------------------


def test_confidence_score_clipped_to_floor_and_ceiling():
    """No single signal should be able to drive the output to 0 or 1."""
    # Worst-case inputs.
    low = confidence_score(face_score=0.0, sharpness=0.0, jitter=1.0)
    assert low >= 0.05
    # Best-case inputs.
    high = confidence_score(face_score=1.0, sharpness=10_000.0, jitter=0.0)
    assert high <= 0.95


def test_confidence_score_monotonic_in_face_score():
    """Higher detector score → higher confidence, all else equal."""
    a = confidence_score(face_score=0.2, sharpness=200.0, jitter=0.3)
    b = confidence_score(face_score=0.8, sharpness=200.0, jitter=0.3)
    assert b > a


def test_confidence_score_monotonic_in_sharpness():
    a = confidence_score(face_score=0.6, sharpness=60.0, jitter=0.0)
    b = confidence_score(face_score=0.6, sharpness=400.0, jitter=0.0)
    assert b > a


def test_confidence_score_jitter_penalises():
    """More jitter → lower confidence (inverted contribution)."""
    a = confidence_score(face_score=0.6, sharpness=200.0, jitter=0.0)
    b = confidence_score(face_score=0.6, sharpness=200.0, jitter=0.9)
    assert b < a


def test_confidence_score_input_clamping():
    """Out-of-range face_score / jitter must not push the output past
    the [floor, ceiling] envelope."""
    s_lo = confidence_score(face_score=-1.0, sharpness=0.0, jitter=2.0)
    s_hi = confidence_score(face_score=10.0, sharpness=10_000.0, jitter=-1.0)
    assert 0.05 <= s_lo <= 0.95
    assert 0.05 <= s_hi <= 0.95


def test_confidence_score_zero_weights_collapses_to_sigmoid_of_bias():
    """When all signal weights are zero, the output is the sigmoid of
    the bias — useful sanity check that the blend formula is correct."""
    import math

    s = confidence_score(
        face_score=1.0,
        sharpness=10_000.0,
        jitter=0.0,
        w_face=0.0,
        w_sharpness=0.0,
        w_jitter=0.0,
        bias=0.0,
        floor=0.0,
        ceiling=1.0,
    )
    # sigmoid(0 * 4) = 0.5
    assert s == pytest.approx(0.5, abs=1e-9)
    s_neg = confidence_score(
        face_score=1.0,
        sharpness=10_000.0,
        jitter=0.0,
        w_face=0.0,
        w_sharpness=0.0,
        w_jitter=0.0,
        bias=-1.0,
        floor=0.0,
        ceiling=1.0,
    )
    expected = 1.0 / (1.0 + math.exp(4.0))  # sigmoid(-1 * 4)
    assert s_neg == pytest.approx(expected, abs=1e-9)
