"""Eye-blink detection from MediaPipe Face Landmarker keypoints (G6).

The Eye Aspect Ratio (EAR, Soukupova & Cech 2016) is the ratio of the
vertical eye opening to the horizontal eye width:

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

…where ``p1..p6`` are six landmarks around one eye in standard order
(corner, upper-1, upper-2, corner, lower-2, lower-1). EAR is ~0.30
when the eye is open and drops below ~0.18 during a blink.

The MediaPipe Face Landmarker (468 landmarks) returns left/right eye
contours from which we pick six points per eye that approximate the
classic dlib 6-point set:

    Right eye (camera's right, user's left):
        outer corner  → 33
        upper-1       → 159
        upper-2       → 158
        inner corner  → 133
        lower-2       → 153
        lower-1       → 145

    Left eye (camera's left, user's right):
        outer corner  → 263
        upper-1       → 386
        upper-2       → 385
        inner corner  → 362
        lower-2       → 380
        lower-1       → 374

A :class:`BlinkDetector` integrates the per-frame EAR over a small
window with hysteresis (require N consecutive frames below threshold
to fire; require EAR to rise above ``threshold + margin`` to clear).
"""

from __future__ import annotations

import collections
import math
from collections.abc import Mapping

#: MediaPipe Face Landmarker indices for the EAR computation, ordered
#: ``(outer, upper1, upper2, inner, lower2, lower1)`` per eye.
RIGHT_EYE_LANDMARKS: tuple[int, int, int, int, int, int] = (33, 159, 158, 133, 153, 145)
LEFT_EYE_LANDMARKS: tuple[int, int, int, int, int, int] = (263, 386, 385, 362, 380, 374)


def eye_aspect_ratio(
    landmarks: Mapping[int, tuple[float, float]],
    eye_indices: tuple[int, int, int, int, int, int],
) -> float | None:
    """Return EAR for one eye, or ``None`` if any landmark is missing.

    Inputs are in pixel coordinates; EAR is scale-invariant, so this
    works equally well on normalised or pixel landmarks as long as
    both are consistent within the call.
    """
    try:
        p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in eye_indices)
    except KeyError:
        return None
    horizontal = _dist(p1, p4)
    if horizontal <= 1e-6:
        return None
    vertical = _dist(p2, p6) + _dist(p3, p5)
    return float(vertical / (2.0 * horizontal))


def mean_ear(
    landmarks: Mapping[int, tuple[float, float]],
) -> float | None:
    """Average EAR across both eyes. ``None`` if either eye's EAR is missing."""
    left = eye_aspect_ratio(landmarks, LEFT_EYE_LANDMARKS)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE_LANDMARKS)
    if left is None or right is None:
        return None
    return 0.5 * (left + right)


class BlinkDetector:
    """Hysteresis-based blink state machine.

    Args:
        closed_threshold:  EAR below this is treated as "eye possibly
                           closed".
        open_margin:       The eye is only considered open again once
                           EAR rises above ``closed_threshold + open_margin``
                           — prevents flicker when EAR hovers around
                           the threshold.
        min_closed_frames: How many consecutive frames must report a
                           closed eye before :attr:`is_blinking` flips
                           to True. Defaults to 2 (~66 ms at 30 fps).
    """

    def __init__(
        self,
        *,
        closed_threshold: float = 0.18,
        open_margin: float = 0.04,
        min_closed_frames: int = 2,
    ) -> None:
        if closed_threshold <= 0 or closed_threshold >= 1:
            raise ValueError(
                f"closed_threshold must be in (0, 1); got {closed_threshold!r}"
            )
        if open_margin < 0:
            raise ValueError(f"open_margin must be ≥ 0; got {open_margin!r}")
        if min_closed_frames < 1:
            raise ValueError(
                f"min_closed_frames must be ≥ 1; got {min_closed_frames!r}"
            )
        self._closed = float(closed_threshold)
        self._open = float(closed_threshold) + float(open_margin)
        self._min_closed = int(min_closed_frames)
        self._closed_streak = 0
        self._is_blinking = False
        # Keep a short EAR history so callers (HUD) can render a sparkline.
        self._history: collections.deque[float] = collections.deque(maxlen=32)

    @property
    def is_blinking(self) -> bool:
        """True while the detector reports a confirmed blink."""
        return self._is_blinking

    @property
    def last_ear(self) -> float | None:
        """Most recently fed EAR value (``None`` before the first sample)."""
        return self._history[-1] if self._history else None

    def reset(self) -> None:
        """Forget streak + history; used on backend stop/restart."""
        self._closed_streak = 0
        self._is_blinking = False
        self._history.clear()

    def update(self, ear: float | None) -> bool:
        """Feed one EAR sample, return the new :attr:`is_blinking` value.

        Passing ``None`` (e.g. eye landmarks unavailable for this frame)
        does not change the state — we assume the eye is in whatever
        state it was last in. This keeps the detector robust to brief
        landmark dropouts mid-blink.
        """
        if ear is None:
            return self._is_blinking
        self._history.append(float(ear))
        if ear < self._closed:
            self._closed_streak += 1
            if self._closed_streak >= self._min_closed:
                self._is_blinking = True
        elif ear >= self._open:
            self._closed_streak = 0
            self._is_blinking = False
        else:
            # In the hysteresis band — keep whatever state we were in.
            # Drop the streak so a hover near the threshold does not
            # eventually fire a false positive.
            self._closed_streak = 0
        return self._is_blinking


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
