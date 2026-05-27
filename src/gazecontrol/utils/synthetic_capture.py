"""Synthetic ``cv2.VideoCapture`` substitute for CI / benchmarking (G11).

The real :class:`cv2.VideoCapture` needs a physical webcam and a desktop
session — neither of which exist on a headless GitHub Actions runner.
:class:`SyntheticVideoCapture` is a drop-in replacement that returns a
deterministic synthetic frame every ``read()`` so the pipeline can be
exercised end-to-end for latency measurements and replay regression
(G12).

Frames are constant 640×480 BGR with a green tint; the value is
arbitrary but stable across reads so any downstream hash / pixel
comparison in tests stays deterministic. Replace the green plane with
a sinusoid if you ever need motion content.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SyntheticVideoCapture:
    """Tiny ``cv2.VideoCapture`` API surface returning fixed BGR frames."""

    def __init__(self, *_args: Any, width: int = 640, height: int = 480, **_kwargs: Any) -> None:
        self._w = int(width)
        self._h = int(height)

    def isOpened(self) -> bool:  # noqa: N802  - mirrors cv2 API
        """Always return True — there is nothing to fail to open."""
        return True

    def set(self, _prop: int, _val: Any) -> bool:
        """Accept any ``CAP_PROP_*`` write; returns True for ergonomics."""
        return True

    def get(self, prop: int) -> float:
        """Return geometry properties; everything else is 0.

        The constants are imported lazily so this module stays importable
        in environments where ``cv2`` is not installed (e.g. headless
        type-check runners).
        """
        try:
            import cv2

            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self._w)
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self._h)
        except ImportError:
            return 0.0
        return 0.0

    def read(self) -> tuple[bool, np.ndarray[Any, Any]]:
        """Return ``(True, BGR frame)`` — never blocks, never fails."""
        frame = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        frame[:, :, 1] = 128  # green channel for sanity-check overlays
        return True, frame

    def release(self) -> None:
        """No-op — there is no underlying device to release."""
        return None


def install_synthetic_capture() -> None:
    """Monkey-patch ``cv2.VideoCapture`` so the pipeline reads synthetics.

    Reversible only by restarting the process; the patch lives for the
    lifetime of the import. Used by ``gazecontrol --benchmark --bench-mock``
    (G11) and the replay harness (G12).
    """
    try:
        import cv2

        cv2.VideoCapture = SyntheticVideoCapture  # type: ignore[assignment, misc]
    except ImportError:
        # cv2 not installed → nothing to patch. The benchmark path will
        # error out shortly anyway when it tries to open a camera.
        pass
