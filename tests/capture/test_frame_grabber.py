"""Tests for FrameGrabber — thread safety, read, recovery."""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest

from tests.helpers import FakeVideoCapture


@pytest.fixture
def grabber():
    """FrameGrabber with patched VideoCapture."""
    from gazecontrol.capture.frame_grabber import FrameGrabber

    with patch("cv2.VideoCapture", return_value=FakeVideoCapture()):
        g = FrameGrabber(camera_index=0, width=640, height=480, fps=30)
        assert g.start() is True
        yield g
        g.stop()


def test_start_stop(grabber):
    # After start(), the running Event should be set.
    assert grabber._running.is_set() is True


def test_read_bgr_returns_frame(grabber):
    time.sleep(0.05)  # let capture loop fill buffer
    ok, frame = grabber.read_bgr()
    assert ok is True
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3


def test_read_bgr_returns_copy_on_repeated_call(grabber):
    """read_bgr should return a copy so the caller can modify it safely."""
    time.sleep(0.05)
    ok1, frame1 = grabber.read_bgr()
    ok2, frame2 = grabber.read_bgr()
    assert ok1 and ok2
    # Both frames should have identical shapes.
    assert frame1.shape == frame2.shape
    assert frame1.dtype == np.uint8


def test_actual_resolution_does_not_block(grabber):
    """actual_resolution must not call _cap.get() from the consumer thread."""
    time.sleep(0.05)
    w, h = grabber.actual_resolution
    assert w == 640
    assert h == 480


def test_no_frame_returns_false():
    from gazecontrol.capture.frame_grabber import FrameGrabber

    g = FrameGrabber()
    # No start() called — _frame is None.
    ok, frame = g.read_bgr()
    assert ok is False
    assert frame is None


def test_camera_drop_handled():
    """After fail_after drops, grabber calls _restart_camera (does not crash)."""
    from gazecontrol.capture.frame_grabber import FrameGrabber

    fake_cap = FakeVideoCapture(fail_after=2)

    restarted = {"called": False}

    def mock_restart(self):
        restarted["called"] = True

    with patch("cv2.VideoCapture", return_value=fake_cap):
        with patch.object(FrameGrabber, "_restart_camera", mock_restart):
            g = FrameGrabber(camera_index=0, width=640, height=480, fps=30)
            g.start()
            time.sleep(0.5)  # enough for 30+ drops at loop speed
            g.stop()

    assert restarted["called"] is True
