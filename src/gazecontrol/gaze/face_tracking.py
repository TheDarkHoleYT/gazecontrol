"""Multi-face disambiguation + identity tracking.

Pure helper functions and a small :class:`FaceTracker` state machine used by
:class:`gazecontrol.gaze.l2cs_backend.L2CSBackend` to pick the *same* face
across frames in scenes with two or more people (G5).

The choice is deliberately tiny — production-grade trackers (SORT,
ByteTrack, DeepSORT) are overkill for desktop gaze control where the
camera is mostly pointed at one user and second faces are intermittent
intruders. We:

1. Score each detection per frame as a linear combination of bounding-box
   area, IoU with the previous frame's locked bbox, and detector
   confidence. The highest-scoring detection wins.
2. Assign a stable :attr:`face_id` when the winner overlaps the previously
   tracked face (IoU above a threshold); otherwise allocate a new id.
3. Emit a :data:`MULTI_FACE` quality flag when more than one detection is
   present so HUD / telemetry consumers know to surface the ambiguity.

All inputs are plain values (tuples, floats) so the module imports
nothing from MediaPipe, ONNX, or NumPy and can be unit-tested under any
environment.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging

logger = logging.getLogger(__name__)

#: Normalised bounding box as (x_min, y_min, x_max, y_max), all in [0, 1].
NormalisedBBox = tuple[float, float, float, float]


@dataclasses.dataclass(frozen=True)
class FaceCandidate:
    """One face detection scored for disambiguation.

    Attributes:
        bbox_norm:  Detection bounding box in normalised frame coords.
        score:      Detector self-reported confidence in [0, 1].
        composite:  Final disambiguation score (higher = preferred), see
                    :func:`score_candidate`.
    """

    bbox_norm: NormalisedBBox
    score: float
    composite: float


def iou(a: NormalisedBBox, b: NormalisedBBox) -> float:
    """Intersection-over-union of two normalised bboxes.

    Returns 0 for disjoint boxes and for degenerate inputs (zero area
    on either side). The function is symmetric and never raises.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _area(bbox: NormalisedBBox) -> float:
    """Normalised area of *bbox*, in [0, 1]."""
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def score_candidate(
    bbox_norm: NormalisedBBox,
    score: float,
    *,
    prev_bbox: NormalisedBBox | None = None,
    w_area: float = 0.4,
    w_iou: float = 0.4,
    w_score: float = 0.2,
) -> float:
    """Composite score balancing size, last-frame proximity, and confidence.

    Weights are calibrated so that:

    * a sole face at the centre always wins,
    * when a previous track exists, a candidate with high IoU beats a
      slightly bigger candidate at the periphery (sticky tracking), and
    * detector self-reported confidence breaks ties at near-equal area
      and proximity.

    All weights are passed in so callers can tune the tradeoff per
    deployment without touching the helper.
    """
    area_component = _area(bbox_norm)
    iou_component = iou(prev_bbox, bbox_norm) if prev_bbox is not None else 0.0
    score_component = max(0.0, min(1.0, score))
    return (
        w_area * area_component
        + w_iou * iou_component
        + w_score * score_component
    )


class FaceTracker:
    """Stable face-id assignment with sticky IoU lock.

    A single :class:`FaceTracker` is owned by the gaze backend for the
    lifetime of a session. Per frame, the backend calls
    :meth:`update` with the list of detection ``(bbox_norm, score)``
    tuples; the tracker returns the chosen face (bbox + id) along with a
    flag indicating whether more than one detection was present this
    frame (used by callers to set ``GazeQuality.MULTI_FACE``).
    """

    def __init__(self, *, lock_iou_threshold: float = 0.3) -> None:
        if not 0.0 < lock_iou_threshold <= 1.0:
            raise ValueError(
                f"lock_iou_threshold must be in (0, 1]; got {lock_iou_threshold!r}"
            )
        self._lock_iou = lock_iou_threshold
        self._next_id = itertools.count(1)
        self._last_bbox: NormalisedBBox | None = None
        self._last_face_id: int | None = None

    @property
    def current_face_id(self) -> int | None:
        """Last assigned face id, or ``None`` before the first detection."""
        return self._last_face_id

    @property
    def current_bbox(self) -> NormalisedBBox | None:
        """Last locked bounding box, or ``None`` before the first detection."""
        return self._last_bbox

    def reset(self) -> None:
        """Drop the locked face — next detection starts a new id sequence."""
        self._last_bbox = None
        self._last_face_id = None

    def update(
        self,
        detections: list[tuple[NormalisedBBox, float]],
        *,
        w_area: float = 0.4,
        w_iou: float = 0.4,
        w_score: float = 0.2,
    ) -> tuple[NormalisedBBox, int, bool] | None:
        """Pick the winning detection and return ``(bbox, face_id, multi)``.

        ``multi`` is True when at least two valid detections were
        present this frame — callers OR :data:`GazeQuality.MULTI_FACE`
        into the prediction's quality flags so the HUD can warn the
        user that another person is being seen.

        Returns ``None`` when *detections* is empty (caller must treat
        the frame as "no face").
        """
        if not detections:
            return None
        scored = [
            FaceCandidate(
                bbox_norm=b,
                score=s,
                composite=score_candidate(
                    b,
                    s,
                    prev_bbox=self._last_bbox,
                    w_area=w_area,
                    w_iou=w_iou,
                    w_score=w_score,
                ),
            )
            for b, s in detections
        ]
        winner = max(scored, key=lambda c: c.composite)
        # Sticky id assignment: keep face_id when the winner overlaps the
        # previously locked face by at least lock_iou_threshold; otherwise
        # the user has likely moved (or a new person took over the frame)
        # — allocate a fresh id.
        if (
            self._last_bbox is not None
            and self._last_face_id is not None
            and iou(self._last_bbox, winner.bbox_norm) >= self._lock_iou
        ):
            face_id = self._last_face_id
        else:
            face_id = next(self._next_id)
            if self._last_face_id is not None:
                logger.debug(
                    "FaceTracker: id rotation %d → %d (iou=%.2f, threshold=%.2f)",
                    self._last_face_id,
                    face_id,
                    iou(self._last_bbox, winner.bbox_norm) if self._last_bbox else 0.0,
                    self._lock_iou,
                )
        self._last_bbox = winner.bbox_norm
        self._last_face_id = face_id
        multi = len(detections) >= 2
        return winner.bbox_norm, face_id, multi
