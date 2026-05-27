"""Tests for the synthetic ``cv2.VideoCapture`` substitute (G11)."""

from __future__ import annotations

import numpy as np

from gazecontrol.utils.synthetic_capture import (
    SyntheticVideoCapture,
    install_synthetic_capture,
)


def test_isopened_always_true():
    cap = SyntheticVideoCapture()
    assert cap.isOpened() is True


def test_read_returns_frame_with_configured_geometry():
    cap = SyntheticVideoCapture(width=320, height=240)
    ok, frame = cap.read()
    assert ok is True
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8


def test_read_frame_has_green_channel_set():
    cap = SyntheticVideoCapture()
    _, frame = cap.read()
    # The green plane should hold the sanity-check value.
    assert int(frame[0, 0, 1]) == 128
    assert int(frame[0, 0, 0]) == 0
    assert int(frame[0, 0, 2]) == 0


def test_get_returns_geometry_via_cv2_props():
    """``get(CAP_PROP_FRAME_WIDTH)`` / ``HEIGHT`` should match the ctor."""
    import cv2

    cap = SyntheticVideoCapture(width=1280, height=720)
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280.0
    assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720.0


def test_get_unknown_prop_returns_zero():
    cap = SyntheticVideoCapture()
    assert cap.get(99_999) == 0.0


def test_set_returns_true_for_any_prop():
    cap = SyntheticVideoCapture()
    assert cap.set(0, 30) is True


def test_release_is_noop():
    cap = SyntheticVideoCapture()
    assert cap.release() is None


def test_install_synthetic_capture_replaces_cv2_factory():
    """After ``install_synthetic_capture`` ``cv2.VideoCapture(...)`` must
    return our synthetic instance, not the real device."""
    import cv2

    original = cv2.VideoCapture
    try:
        install_synthetic_capture()
        cap = cv2.VideoCapture(0)
        assert isinstance(cap, SyntheticVideoCapture)
        # Round-trip a read through the patched factory.
        ok, frame = cap.read()
        assert ok and frame.shape == (480, 640, 3)
    finally:
        cv2.VideoCapture = original
