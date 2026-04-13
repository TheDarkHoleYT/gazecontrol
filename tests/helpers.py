"""Shared test helpers — importable from any test file."""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Fake VideoCapture for FrameGrabber tests
# ---------------------------------------------------------------------------


class FakeVideoCapture:
    """Minimal cv2.VideoCapture substitute for tests."""

    def __init__(
        self,
        *args: object,
        width: int = 640,
        height: int = 480,
        fail_after: int = -1,
    ) -> None:
        self._open = True
        self._width = width
        self._height = height
        self._read_count = 0
        self._fail_after = fail_after

    def isOpened(self) -> bool:
        return self._open

    def set(self, prop: int, val: object) -> bool:
        return True

    def get(self, prop: int) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        self._read_count += 1
        if self._fail_after >= 0 and self._read_count > self._fail_after:
            return False, None
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame[:, :, 1] = 128  # green channel for easy identification
        return True, frame

    def release(self) -> None:
        self._open = False


# ---------------------------------------------------------------------------
# Fake hand result (MediaPipe compatibility wrapper)
# ---------------------------------------------------------------------------


def make_fake_hand_result(
    n_hands: int = 1,
    pinch: bool = False,
) -> object:
    """Create a MediaPipe-compatible hand result for gesture tests."""

    class _LM:
        def __init__(self, x: float, y: float, z: float = 0.0) -> None:
            self.x = x
            self.y = y
            self.z = z

    class _Landmarks:
        def __init__(self) -> None:
            self.landmark = [_LM(0.5, 0.5) for _ in range(21)]
            if pinch:
                self.landmark[4] = _LM(0.5, 0.5)
                self.landmark[8] = _LM(0.5, 0.5)

    class _Result:
        def __init__(self) -> None:
            self.multi_hand_landmarks = [_Landmarks() for _ in range(n_hands)]
            self.multi_handedness = [None] * n_hands

    return _Result()
