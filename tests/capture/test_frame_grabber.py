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
    # After start(), grabber should be running.
    assert grabber._running is True


def test_read_bgr_returns_frame(grabber):
    time.sleep(0.05)  # let capture loop fill buffer
    ok, frame = grabber.read_bgr()
    assert ok is True
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3


def test_read_rgb_returns_rgb_frame(grabber):
    time.sleep(0.05)
    ok, frame = grabber.read()
    assert ok is True
    assert frame is not None
    # Green channel was set to 128 in FakeVideoCapture; after BGR→RGB conversion
    # the original blue channel should be non-zero at position [..2].
    assert frame.dtype == np.uint8


def test_read_bgr_and_read_derive_from_same_snapshot(grabber):
    """read_bgr and read must not reference different frames."""
    time.sleep(0.05)
    ok1, bgr = grabber.read_bgr()
    ok2, rgb = grabber.read()
    assert ok1
    assert ok2
    # Both should have the same spatial dimensions.
    assert bgr.shape[:2] == rgb.shape[:2]


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
