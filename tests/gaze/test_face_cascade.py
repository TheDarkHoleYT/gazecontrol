"""Tests for the face-detection cascade (G4)."""

from __future__ import annotations

import pytest

from gazecontrol.gaze.face_cascade import (
    FaceDetectionCascade,
    bbox_from_landmarks_norm,
)

# ---------------------------------------------------------------------------
# bbox_from_landmarks_norm
# ---------------------------------------------------------------------------


def test_bbox_from_landmarks_returns_extents_with_padding():
    lm = {0: (0.3, 0.3), 1: (0.7, 0.5), 2: (0.4, 0.8)}
    bbox = bbox_from_landmarks_norm(lm, padding=0.1)
    assert bbox is not None
    x_min, y_min, x_max, y_max = bbox
    # extent before padding: x∈[0.3, 0.7] (w=0.4), y∈[0.3, 0.8] (h=0.5)
    # padded by 10 %: pad_x = 0.04, pad_y = 0.05
    assert x_min == pytest.approx(0.3 - 0.04)
    assert y_min == pytest.approx(0.3 - 0.05)
    assert x_max == pytest.approx(0.7 + 0.04)
    assert y_max == pytest.approx(0.8 + 0.05)


def test_bbox_from_landmarks_clamps_to_unit_square():
    lm = {0: (0.0, 0.0), 1: (1.0, 1.0)}
    bbox = bbox_from_landmarks_norm(lm, padding=0.2)
    assert bbox is not None
    assert bbox == (0.0, 0.0, 1.0, 1.0)


def test_bbox_from_landmarks_returns_none_on_empty_dict():
    assert bbox_from_landmarks_norm({}) is None


def test_bbox_from_landmarks_returns_none_on_collapsed_point():
    lm = {0: (0.5, 0.5), 1: (0.5, 0.5)}
    assert bbox_from_landmarks_norm(lm) is None


def test_bbox_from_landmarks_default_padding_is_5_percent():
    lm = {0: (0.4, 0.4), 1: (0.6, 0.6)}
    bbox = bbox_from_landmarks_norm(lm)
    assert bbox is not None
    # width 0.2 → pad 0.01.
    assert bbox[0] == pytest.approx(0.39)
    assert bbox[2] == pytest.approx(0.61)


# ---------------------------------------------------------------------------
# FaceDetectionCascade
# ---------------------------------------------------------------------------


def test_cascade_blaze_wins_when_present():
    c = FaceDetectionCascade(max_replay_frames=5)
    out = c.step(blaze_bbox=(0.1, 0.1, 0.4, 0.4), landmarker_bbox=(0.0, 0.0, 0.9, 0.9))
    assert out is not None
    assert out.tier == "blaze"
    assert out.bbox == (0.1, 0.1, 0.4, 0.4)
    assert c.telemetry() == {"blaze": 1, "landmarker": 0, "replay": 0}


def test_cascade_falls_back_to_landmarker_when_blaze_misses():
    c = FaceDetectionCascade()
    out = c.step(blaze_bbox=None, landmarker_bbox=(0.0, 0.0, 0.5, 0.5))
    assert out is not None
    assert out.tier == "landmarker"
    assert out.bbox == (0.0, 0.0, 0.5, 0.5)


def test_cascade_replays_last_bbox_when_both_detectors_miss():
    c = FaceDetectionCascade(max_replay_frames=3)
    # Frame 1: Blaze hits.
    c.step(blaze_bbox=(0.2, 0.2, 0.6, 0.6), landmarker_bbox=None)
    # Frame 2: both miss → replay.
    out = c.step(blaze_bbox=None, landmarker_bbox=None)
    assert out is not None
    assert out.tier == "replay"
    assert out.bbox == (0.2, 0.2, 0.6, 0.6)
    assert out.replay_frames_used == 1


def test_cascade_replay_resets_after_a_real_detection():
    c = FaceDetectionCascade(max_replay_frames=3)
    c.step(blaze_bbox=(0.1, 0.1, 0.5, 0.5), landmarker_bbox=None)
    c.step(blaze_bbox=None, landmarker_bbox=None)  # replay 1
    c.step(blaze_bbox=None, landmarker_bbox=None)  # replay 2
    assert c.replay_streak == 2
    out = c.step(blaze_bbox=(0.2, 0.2, 0.6, 0.6), landmarker_bbox=None)
    assert out is not None and out.tier == "blaze"
    assert c.replay_streak == 0


def test_cascade_gives_up_after_max_replay_frames():
    c = FaceDetectionCascade(max_replay_frames=2)
    c.step(blaze_bbox=(0.1, 0.1, 0.5, 0.5), landmarker_bbox=None)
    c.step(blaze_bbox=None, landmarker_bbox=None)  # replay 1
    c.step(blaze_bbox=None, landmarker_bbox=None)  # replay 2
    # Replay budget exhausted — None must surface so the backend knows.
    out = c.step(blaze_bbox=None, landmarker_bbox=None)
    assert out is None
    # And state has been cleared so a fresh detection starts a new sequence.
    assert c.last_bbox is None


def test_cascade_max_replay_zero_disables_replay():
    c = FaceDetectionCascade(max_replay_frames=0)
    c.step(blaze_bbox=(0.1, 0.1, 0.5, 0.5), landmarker_bbox=None)
    out = c.step(blaze_bbox=None, landmarker_bbox=None)
    assert out is None


def test_cascade_negative_max_replay_rejected():
    with pytest.raises(ValueError):
        FaceDetectionCascade(max_replay_frames=-1)


def test_cascade_returns_none_when_no_signal_ever_landed():
    c = FaceDetectionCascade(max_replay_frames=3)
    assert c.step(blaze_bbox=None, landmarker_bbox=None) is None


def test_cascade_telemetry_counts_per_tier():
    c = FaceDetectionCascade(max_replay_frames=5)
    c.step(blaze_bbox=(0, 0, 0.5, 0.5), landmarker_bbox=None)
    c.step(blaze_bbox=None, landmarker_bbox=(0, 0, 0.5, 0.5))
    c.step(blaze_bbox=None, landmarker_bbox=None)
    c.step(blaze_bbox=None, landmarker_bbox=None)
    counts = c.telemetry()
    assert counts == {"blaze": 1, "landmarker": 1, "replay": 2}


def test_cascade_reset_clears_state_and_counts():
    c = FaceDetectionCascade()
    c.step(blaze_bbox=(0, 0, 0.5, 0.5), landmarker_bbox=None)
    c.reset()
    assert c.last_bbox is None
    assert c.replay_streak == 0
    assert c.telemetry() == {"blaze": 0, "landmarker": 0, "replay": 0}
