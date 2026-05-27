"""Face-detection cascade (G4).

The v0.7/v0.8 pipeline trusted a single BlazeFace detection per frame
and, when that failed, fell back to a brittle centre-frame crop. For
v1.0 we run a cascade:

1. **BlazeFace primary** — fast and accurate when the face is roughly
   frontal and unoccluded. Already wired in :mod:`l2cs_backend`.
2. **Face Landmarker bounding box** — derived from the 468-landmark
   mesh produced by the same Face Landmarker we already run for
   head-pose PnP / EAR blink (G2, G6). Heavier than BlazeFace but
   recovers gracefully when partial occlusions (hair, glasses) trip
   the BlazeFace classifier head.
3. **Previous-frame replay** — when both detectors miss this frame,
   reuse the last known bbox up to ``max_replay_frames`` later. Cheap
   sticky-tracking for brief detector hiccups; never used past a hard
   limit so a vanished face is eventually reported as missing.

The cascade is a *pure* state machine — it never touches MediaPipe or
ONNX itself. The backend feeds in whichever detections it has and
gets back the picked bounding box plus the tier responsible for the
choice (handy for telemetry / HUD: "fell back to landmarks").
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import Literal

from gazecontrol.gaze.face_tracking import NormalisedBBox

logger = logging.getLogger(__name__)

#: Which tier of the cascade produced the winning bbox.
CascadeTier = Literal["blaze", "landmarker", "replay"]


@dataclasses.dataclass(frozen=True)
class CascadeOutcome:
    """The cascade's per-frame result.

    Attributes:
        bbox:               Winning bounding box in normalised
                            frame coords ``(x_min, y_min, x_max, y_max)``.
        tier:               Which detector tier produced the bbox.
        replay_frames_used: How many consecutive frames have now been
                            served from the replay tier. Always 0
                            when ``tier != "replay"``.
    """

    bbox: NormalisedBBox
    tier: CascadeTier
    replay_frames_used: int = 0


def bbox_from_landmarks_norm(
    landmarks_norm: Mapping[int, tuple[float, float]],
    *,
    padding: float = 0.05,
) -> NormalisedBBox | None:
    """Derive a normalised bbox from the (min, max) extents of *landmarks_norm*.

    Args:
        landmarks_norm:  Mapping ``landmark_id → (x_norm, y_norm)``.
                         Each coord must already be in ``[0, 1]``.
        padding:         Fraction of bbox width/height to add on each
                         side so the crop survives slight extrapolation
                         beyond the landmark extent. Defaults to 5 %.

    Returns ``None`` when the landmark dict is empty or degenerates
    to a single point.
    """
    if not landmarks_norm:
        return None
    xs = [p[0] for p in landmarks_norm.values()]
    ys = [p[1] for p in landmarks_norm.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max <= x_min or y_max <= y_min:
        return None
    w = x_max - x_min
    h = y_max - y_min
    pad_x = padding * w
    pad_y = padding * h
    return (
        max(0.0, x_min - pad_x),
        max(0.0, y_min - pad_y),
        min(1.0, x_max + pad_x),
        min(1.0, y_max + pad_y),
    )


class FaceDetectionCascade:
    """Three-tier face-detection state machine.

    Args:
        max_replay_frames: Hard cap on how long the cascade may serve
                           a stale bbox before declaring the face
                           missing. ``0`` disables replay entirely.
    """

    def __init__(self, *, max_replay_frames: int = 5) -> None:
        if max_replay_frames < 0:
            raise ValueError(
                f"max_replay_frames must be ≥ 0; got {max_replay_frames!r}"
            )
        self._max_replay = int(max_replay_frames)
        self._last_bbox: NormalisedBBox | None = None
        self._replay_streak = 0
        self._tier_counts: dict[CascadeTier, int] = {
            "blaze": 0,
            "landmarker": 0,
            "replay": 0,
        }

    @property
    def last_bbox(self) -> NormalisedBBox | None:
        """Most recently returned bbox, or ``None`` before the first hit."""
        return self._last_bbox

    @property
    def replay_streak(self) -> int:
        """How many consecutive frames have come from the replay tier."""
        return self._replay_streak

    def telemetry(self) -> dict[str, int]:
        """Cumulative hit counts per tier, for logging / Prometheus."""
        return {str(k): v for k, v in self._tier_counts.items()}

    def reset(self) -> None:
        """Forget all state — called by the backend on stop/restart."""
        self._last_bbox = None
        self._replay_streak = 0
        for k in self._tier_counts:
            self._tier_counts[k] = 0

    def step(
        self,
        *,
        blaze_bbox: NormalisedBBox | None,
        landmarker_bbox: NormalisedBBox | None,
    ) -> CascadeOutcome | None:
        """Pick a bbox for the current frame.

        Args:
            blaze_bbox:        Tier 1 — output of the BlazeFace
                               detector (after multi-face
                               disambiguation, G5). ``None`` when
                               BlazeFace returned no usable
                               detection.
            landmarker_bbox:   Tier 2 — bbox derived from the Face
                               Landmarker mesh via
                               :func:`bbox_from_landmarks_norm`.
                               ``None`` when the landmarker is
                               disabled or returned no face.

        Returns the winning bbox + tier label, or ``None`` when all
        three tiers fail (BlazeFace miss + Landmarker miss + replay
        budget exhausted).
        """
        if blaze_bbox is not None:
            self._last_bbox = blaze_bbox
            self._replay_streak = 0
            self._tier_counts["blaze"] += 1
            return CascadeOutcome(bbox=blaze_bbox, tier="blaze")
        if landmarker_bbox is not None:
            self._last_bbox = landmarker_bbox
            self._replay_streak = 0
            self._tier_counts["landmarker"] += 1
            return CascadeOutcome(bbox=landmarker_bbox, tier="landmarker")
        if self._last_bbox is not None and self._replay_streak < self._max_replay:
            self._replay_streak += 1
            self._tier_counts["replay"] += 1
            return CascadeOutcome(
                bbox=self._last_bbox,
                tier="replay",
                replay_frames_used=self._replay_streak,
            )
        # All tiers exhausted — caller treats this as "no face".
        self._replay_streak = 0
        self._last_bbox = None
        return None
