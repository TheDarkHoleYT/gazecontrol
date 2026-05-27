"""Tests for the EAR-based blink detector (G6)."""

from __future__ import annotations

import pytest

from gazecontrol.gaze.blink import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    BlinkDetector,
    eye_aspect_ratio,
    mean_ear,
)

# ---------------------------------------------------------------------------
# eye_aspect_ratio / mean_ear
# ---------------------------------------------------------------------------


def _open_eye_landmarks(eye_indices):
    """Synthetic landmarks for an "open" eye (vertical opening 8 px,
    horizontal width 30 px, EAR ≈ 8/30 ≈ 0.267)."""
    cx, cy = 100.0, 100.0
    # outer / inner corners spaced 30 px apart, upper/lower pairs at ±4 px.
    return {
        eye_indices[0]: (cx - 15.0, cy),
        eye_indices[1]: (cx - 5.0, cy - 4.0),
        eye_indices[2]: (cx + 5.0, cy - 4.0),
        eye_indices[3]: (cx + 15.0, cy),
        eye_indices[4]: (cx + 5.0, cy + 4.0),
        eye_indices[5]: (cx - 5.0, cy + 4.0),
    }


def _closed_eye_landmarks(eye_indices):
    """Synthetic landmarks for a "closed" eye (vertical opening 1 px,
    horizontal width 30 px, EAR ≈ 1/30 ≈ 0.033)."""
    cx, cy = 100.0, 100.0
    return {
        eye_indices[0]: (cx - 15.0, cy),
        eye_indices[1]: (cx - 5.0, cy - 0.5),
        eye_indices[2]: (cx + 5.0, cy - 0.5),
        eye_indices[3]: (cx + 15.0, cy),
        eye_indices[4]: (cx + 5.0, cy + 0.5),
        eye_indices[5]: (cx - 5.0, cy + 0.5),
    }


def test_eye_aspect_ratio_open_eye_around_typical_value():
    lm = _open_eye_landmarks(RIGHT_EYE_LANDMARKS)
    ear = eye_aspect_ratio(lm, RIGHT_EYE_LANDMARKS)
    assert ear is not None
    assert ear == pytest.approx(8.0 / 30.0, rel=1e-6)


def test_eye_aspect_ratio_closed_eye_far_below_threshold():
    lm = _closed_eye_landmarks(RIGHT_EYE_LANDMARKS)
    ear = eye_aspect_ratio(lm, RIGHT_EYE_LANDMARKS)
    assert ear is not None
    assert ear < 0.10


def test_eye_aspect_ratio_returns_none_on_missing_landmark():
    lm = _open_eye_landmarks(RIGHT_EYE_LANDMARKS)
    lm.pop(33)  # drop the outer corner
    assert eye_aspect_ratio(lm, RIGHT_EYE_LANDMARKS) is None


def test_eye_aspect_ratio_returns_none_on_zero_width():
    lm = _open_eye_landmarks(RIGHT_EYE_LANDMARKS)
    lm[33] = lm[133]  # collapse the two corners to the same point
    assert eye_aspect_ratio(lm, RIGHT_EYE_LANDMARKS) is None


def test_mean_ear_averages_both_eyes():
    lm = {}
    lm.update(_open_eye_landmarks(RIGHT_EYE_LANDMARKS))
    lm.update(_open_eye_landmarks(LEFT_EYE_LANDMARKS))
    m = mean_ear(lm)
    assert m is not None
    assert m == pytest.approx(8.0 / 30.0, rel=1e-6)


def test_mean_ear_returns_none_when_one_eye_missing():
    lm = _open_eye_landmarks(RIGHT_EYE_LANDMARKS)
    assert mean_ear(lm) is None


# ---------------------------------------------------------------------------
# BlinkDetector
# ---------------------------------------------------------------------------


def test_blink_detector_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        BlinkDetector(closed_threshold=0.0)
    with pytest.raises(ValueError):
        BlinkDetector(closed_threshold=1.5)


def test_blink_detector_invalid_margin_or_frames_rejected():
    with pytest.raises(ValueError):
        BlinkDetector(open_margin=-0.01)
    with pytest.raises(ValueError):
        BlinkDetector(min_closed_frames=0)


def test_blink_detector_fires_after_min_closed_frames():
    d = BlinkDetector(closed_threshold=0.2, open_margin=0.05, min_closed_frames=2)
    assert d.update(0.30) is False  # open
    assert d.update(0.15) is False  # one closed frame — not yet
    assert d.update(0.10) is True  # second closed frame → fire


def test_blink_detector_clears_only_above_open_margin():
    d = BlinkDetector(closed_threshold=0.2, open_margin=0.05, min_closed_frames=1)
    d.update(0.10)
    assert d.is_blinking is True
    # EAR rises into the hysteresis band — still blinking.
    assert d.update(0.22) is True
    # EAR rises past closed + margin = 0.25 → cleared.
    assert d.update(0.30) is False


def test_blink_detector_streak_resets_on_unambiguously_open_frame():
    """A clean open frame must reset the streak, not just clear the flag."""
    d = BlinkDetector(closed_threshold=0.2, open_margin=0.05, min_closed_frames=3)
    d.update(0.15)
    d.update(0.15)  # two closed — not yet firing
    d.update(0.30)  # open — streak reset
    # Two more closed frames should not be enough now.
    assert d.update(0.15) is False
    assert d.update(0.15) is False


def test_blink_detector_none_input_preserves_state():
    """Missing landmarks (None) must not flip the state — preserves
    robustness across brief landmark dropouts mid-blink."""
    d = BlinkDetector(closed_threshold=0.2, min_closed_frames=1)
    d.update(0.10)
    assert d.is_blinking is True
    assert d.update(None) is True  # still blinking
    assert d.update(0.05) is True


def test_blink_detector_last_ear_tracks_input():
    d = BlinkDetector()
    assert d.last_ear is None
    d.update(0.27)
    assert d.last_ear == pytest.approx(0.27)


def test_blink_detector_reset_clears_state():
    d = BlinkDetector(min_closed_frames=1)
    d.update(0.10)
    assert d.is_blinking is True
    d.reset()
    assert d.is_blinking is False
    assert d.last_ear is None
