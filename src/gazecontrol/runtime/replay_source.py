"""Replay frame source for deterministic regression testing (G12).

The runtime pipeline normally reads frames from :class:`FrameGrabber`
(threaded webcam). For CI regression testing and gaze-accuracy
benchmarking we instead drive the pipeline from a recorded clip + a
sidecar ground-truth file::

    tests/fixtures/replay_short/
        clip.mp4
        ground_truth.jsonl   one JSON object per line:
            {"frame_id": 0, "gaze_x": 960, "gaze_y": 540}
            {"frame_id": 1, "gaze_x": 962, "gaze_y": 543}
            ...

:class:`ReplayFrameSource` exposes the same ``read_bgr()`` contract as
:class:`FrameGrabber` so :class:`CaptureStage` swaps it transparently.
The ground-truth lookup is exposed separately so a wrapper stage can
compute the per-frame Euclidean error and feed it back into the
profiler (gauge ``gazecontrol_gaze_error_px``).

The harness is intentionally library-only — wiring it into the live
pipeline via a ``--replay`` CLI flag is a follow-up commit so this
module ships green without forcing the rest of the runtime to grow a
``frame_source`` indirection prematurely.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class GroundTruth:
    """Per-frame ground-truth gaze loaded from a ``.jsonl`` sidecar.

    Each line is a JSON object with at least:

    * ``frame_id`` (int) — 0-based.
    * ``gaze_x``, ``gaze_y`` (int | float, pixels) — the *correct*
      screen point for that frame.

    Missing frames are allowed (the harness simply skips them in the
    accuracy comparison). Duplicate ``frame_id`` entries override the
    previous one.
    """

    def __init__(self, samples: dict[int, tuple[float, float]]) -> None:
        self._samples = dict(samples)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> GroundTruth:
        """Load from a one-object-per-line JSON file.

        Lines that fail to parse or that omit any of the three required
        keys are skipped with a DEBUG log message — the harness should
        never crash on a malformed fixture.
        """
        samples: dict[int, tuple[float, float]] = {}
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open("r", encoding="utf-8") as fh:
            for n, raw in enumerate(fh):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.debug("GroundTruth: line %d invalid JSON: %s", n, exc)
                    continue
                try:
                    fid = int(obj["frame_id"])
                    gx = float(obj["gaze_x"])
                    gy = float(obj["gaze_y"])
                except (KeyError, TypeError, ValueError) as exc:
                    logger.debug("GroundTruth: line %d missing keys: %s", n, exc)
                    continue
                samples[fid] = (gx, gy)
        return cls(samples)

    def __len__(self) -> int:
        """Number of ground-truth samples loaded."""
        return len(self._samples)

    def __contains__(self, frame_id: int) -> bool:
        """True when a ground-truth sample exists for *frame_id*."""
        return frame_id in self._samples

    def expected(self, frame_id: int) -> tuple[float, float] | None:
        """Return ``(gx, gy)`` for *frame_id*, or ``None`` when unknown."""
        return self._samples.get(int(frame_id))

    def error_px(
        self,
        frame_id: int,
        predicted_xy: tuple[float, float] | None,
    ) -> float | None:
        """Euclidean distance between prediction and ground truth.

        Returns ``None`` when either the prediction or the ground truth
        is missing — callers treat that as "no measurement for this
        frame" rather than scoring a zero error.
        """
        if predicted_xy is None:
            return None
        gt = self.expected(frame_id)
        if gt is None:
            return None
        dx = predicted_xy[0] - gt[0]
        dy = predicted_xy[1] - gt[1]
        return float((dx * dx + dy * dy) ** 0.5)


class ReplayFrameSource:
    """Drop-in frame source backed by a recorded clip.

    Mirrors the :meth:`FrameGrabber.read_bgr` contract so the capture
    stage can substitute it without touching its own internals::

        ok, frame_bgr = source.read_bgr()

    Args:
        video_path:         ``.mp4`` (or anything ``cv2.VideoCapture``
                            can open) with the recorded frames.
        ground_truth_path:  Optional ``.jsonl`` sidecar. When set, the
                            consumer can look up the expected gaze for
                            each frame via :meth:`expected_gaze`.
        loop:               When True, re-open the clip at EOF so the
                            harness can drive longer benchmarks against
                            a short clip. Defaults to False (single pass).
    """

    def __init__(
        self,
        video_path: str | Path,
        ground_truth_path: str | Path | None = None,
        *,
        loop: bool = False,
    ) -> None:
        self._video_path = Path(video_path)
        self._gt_path = Path(ground_truth_path) if ground_truth_path else None
        self._loop = bool(loop)
        self._cap: Any = None
        self._frame_id: int = 0
        self._gt: GroundTruth | None = None
        if not self._video_path.exists():
            raise FileNotFoundError(self._video_path)

    def start(self) -> bool:
        """Open the video file. Returns True iff cv2 + the file cooperate."""
        try:
            import cv2
        except ImportError:
            logger.exception("ReplayFrameSource: cv2 unavailable.")
            return False
        self._cap = cv2.VideoCapture(str(self._video_path))
        if not self._cap.isOpened():
            logger.error("ReplayFrameSource: failed to open %s", self._video_path)
            return False
        if self._gt_path is not None:
            try:
                self._gt = GroundTruth.from_jsonl(self._gt_path)
            except (OSError, json.JSONDecodeError):
                logger.exception(
                    "ReplayFrameSource: failed to parse ground-truth %s", self._gt_path
                )
                self._gt = None
        return True

    def stop(self) -> None:
        """Release the underlying capture handle. Idempotent."""
        if self._cap is not None:
            try:
                self._cap.release()
            except (RuntimeError, OSError):
                logger.debug("ReplayFrameSource: cap.release raised.", exc_info=True)
            self._cap = None

    @property
    def frame_id(self) -> int:
        """0-based index of the next frame to be read."""
        return self._frame_id

    @property
    def ground_truth(self) -> GroundTruth | None:
        """The parsed ground-truth side-car, or None when none was provided."""
        return self._gt

    def expected_gaze(self, frame_id: int | None = None) -> tuple[float, float] | None:
        """Return the ground-truth gaze for *frame_id*.

        Defaults to the most recently read frame when *frame_id* is None.
        """
        if self._gt is None:
            return None
        fid = frame_id if frame_id is not None else max(0, self._frame_id - 1)
        return self._gt.expected(fid)

    def read_bgr(self) -> tuple[bool, np.ndarray[Any, Any] | None]:
        """Return ``(ok, frame_bgr)`` matching :class:`FrameGrabber`.

        At EOF the source either rewinds (when ``loop=True``) or returns
        ``(False, None)`` once and stays at EOF for subsequent calls.
        """
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            if self._loop:
                self._cap.set(__import__("cv2").CAP_PROP_POS_FRAMES, 0)
                self._frame_id = 0
                ok, frame = self._cap.read()
                if not ok:
                    return False, None
            else:
                return False, None
        self._frame_id += 1
        return True, frame
