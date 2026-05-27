"""Tests for the replay frame source + ground-truth helpers (G12)."""

from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from gazecontrol.runtime.replay_source import GroundTruth, ReplayFrameSource  # noqa: E402


def _write_mp4(path, n_frames: int = 5) -> None:
    """Write a tiny ``.mp4`` so the replay source has something to open."""
    h, w = 120, 160
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Encode the frame index as the blue-channel value for sanity.
        frame[:, :, 0] = i * 50
        writer.write(frame)
    writer.release()


def _write_jsonl(path, samples) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# GroundTruth
# ---------------------------------------------------------------------------


def test_ground_truth_loads_well_formed_lines(tmp_path):
    p = tmp_path / "gt.jsonl"
    _write_jsonl(
        p,
        [
            {"frame_id": 0, "gaze_x": 100, "gaze_y": 200},
            {"frame_id": 1, "gaze_x": 110.5, "gaze_y": 205.0},
        ],
    )
    gt = GroundTruth.from_jsonl(p)
    assert len(gt) == 2
    assert gt.expected(0) == (100.0, 200.0)
    assert gt.expected(1) == (110.5, 205.0)


def test_ground_truth_skips_invalid_lines(tmp_path):
    p = tmp_path / "gt.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write('{"frame_id": 0, "gaze_x": 0, "gaze_y": 0}\n')
        fh.write("not-json\n")  # malformed
        fh.write('{"frame_id": 1}\n')  # missing keys
        fh.write("\n")  # blank
        fh.write('{"frame_id": 2, "gaze_x": 5, "gaze_y": 6}\n')
    gt = GroundTruth.from_jsonl(p)
    assert len(gt) == 2
    assert gt.expected(0) == (0.0, 0.0)
    assert gt.expected(2) == (5.0, 6.0)


def test_ground_truth_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        GroundTruth.from_jsonl(tmp_path / "nope.jsonl")


def test_ground_truth_error_px_handles_missing_inputs(tmp_path):
    p = tmp_path / "gt.jsonl"
    _write_jsonl(p, [{"frame_id": 0, "gaze_x": 0, "gaze_y": 0}])
    gt = GroundTruth.from_jsonl(p)
    # Known frame, known prediction.
    assert gt.error_px(0, (3.0, 4.0)) == pytest.approx(5.0)
    # Missing prediction.
    assert gt.error_px(0, None) is None
    # Missing ground truth.
    assert gt.error_px(99, (0.0, 0.0)) is None


def test_ground_truth_membership_and_iter(tmp_path):
    p = tmp_path / "gt.jsonl"
    _write_jsonl(
        p,
        [
            {"frame_id": 7, "gaze_x": 0, "gaze_y": 0},
            {"frame_id": 9, "gaze_x": 1, "gaze_y": 1},
        ],
    )
    gt = GroundTruth.from_jsonl(p)
    assert 7 in gt
    assert 8 not in gt


# ---------------------------------------------------------------------------
# ReplayFrameSource
# ---------------------------------------------------------------------------


def test_replay_source_iterates_frames(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_mp4(video, n_frames=4)
    src = ReplayFrameSource(video)
    assert src.start() is True
    frame_ids = []
    while True:
        ok, frame = src.read_bgr()
        if not ok:
            break
        assert frame is not None
        frame_ids.append(src.frame_id)
    src.stop()
    assert frame_ids[:4] == [1, 2, 3, 4]


def test_replay_source_eof_returns_false_without_loop(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_mp4(video, n_frames=2)
    src = ReplayFrameSource(video)
    src.start()
    src.read_bgr()
    src.read_bgr()
    ok, frame = src.read_bgr()
    assert ok is False
    assert frame is None


def test_replay_source_loop_rewinds_at_eof(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_mp4(video, n_frames=2)
    src = ReplayFrameSource(video, loop=True)
    src.start()
    src.read_bgr()
    src.read_bgr()
    ok, frame = src.read_bgr()
    assert ok is True
    assert frame is not None
    # frame_id resets to 1 on the first frame after rewind.
    assert src.frame_id == 1


def test_replay_source_ground_truth_lookup(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_mp4(video, n_frames=2)
    gt_path = tmp_path / "gt.jsonl"
    _write_jsonl(
        gt_path,
        [
            {"frame_id": 0, "gaze_x": 10, "gaze_y": 20},
            {"frame_id": 1, "gaze_x": 30, "gaze_y": 40},
        ],
    )
    src = ReplayFrameSource(video, gt_path)
    src.start()
    src.read_bgr()
    # Most recently read frame is frame_id 1 → src.frame_id == 1 →
    # expected_gaze() defaults to frame_id 0 (max(0, frame_id-1)).
    assert src.expected_gaze() == (10.0, 20.0)
    src.read_bgr()
    assert src.expected_gaze() == (30.0, 40.0)


def test_replay_source_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReplayFrameSource(tmp_path / "missing.mp4")


def test_replay_source_stop_is_idempotent(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_mp4(video, n_frames=2)
    src = ReplayFrameSource(video)
    src.start()
    src.stop()
    src.stop()  # second call must not raise
