"""FrameGrabber — threaded webcam reader with auto-recovery.

Thread-safety:
    All access to ``self._cap`` is serialised under ``_cap_lock``.
    The capture loop holds the lock only while reading; restart holds
    it while replacing ``_cap`` so consumers see a consistent object.

    ``actual_resolution`` reads from a cached snapshot updated inside
    the capture loop — it never calls ``_cap.get()`` from the consumer
    thread (avoids concurrent device access).
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time

import cv2
import numpy as np

from gazecontrol.settings import get_settings

logger = logging.getLogger(__name__)


class FrameGrabber:
    """Capture webcam frames in a background thread; always expose the latest frame.

    Usage::

        grabber = FrameGrabber()
        if grabber.start():
            ok, frame_bgr = grabber.read_bgr()
            ...
        grabber.stop()
    """

    def __init__(
        self,
        camera_index: int | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> None:
        s = get_settings().camera
        self.camera_index = camera_index if camera_index is not None else s.index
        self.width = width if width is not None else s.width
        self.height = height if height is not None else s.height
        self.fps = fps if fps is not None else s.fps

        self._cap: cv2.VideoCapture | None = None
        self._cap_lock = threading.Lock()

        self._frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._frame_count: int = 0

        # Cached resolution snapshot — updated by capture loop, never by consumer.
        self._actual_w: int = 0
        self._actual_h: int = 0

        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Open the camera and start the background capture thread.

        Returns:
            True on success, False if the camera cannot be opened.
        """
        with self._cap_lock:
            cap = self._open_cap(self.camera_index)
            if cap is None:
                return False

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w != self.width or actual_h != self.height:
                logger.warning(
                    "Requested %dx%d not supported; using %dx%d.",
                    self.width, self.height, actual_w, actual_h,
                )
                self.width = actual_w
                self.height = actual_h
            self._actual_w = actual_w
            self._actual_h = actual_h

            # Warm-up read (OpenCV may return black frames at first).
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera opened but first read failed.")
                cap.release()
                return False

            self._cap = cap

        with self._frame_lock:
            self._frame = frame

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("FrameGrabber started: %dx%d@%dfps", self.width, self.height, self.fps)
        return True

    def _open_cap(self, index: int) -> cv2.VideoCapture | None:
        """Try to open a VideoCapture with DirectShow fallback."""
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                return cap
            cap.release()
        logger.error("Cannot open camera index=%d (tried CAP_DSHOW and CAP_ANY).", index)
        return None

    # ------------------------------------------------------------------
    # Capture loop (background thread)
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Read frames continuously, refreshing the shared frame buffer."""
        consecutive_drops = 0
        max_drops = 30  # ~1 s at 30 fps before restart attempt

        while self._running:
            with self._cap_lock:
                cap = self._cap

            if cap is None:
                time.sleep(0.033)
                continue

            ret, frame = cap.read()
            if ret:
                consecutive_drops = 0
                # Cache resolution from within the loop (no consumer-side device call).
                self._actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                with self._frame_lock:
                    self._frame = frame
                    self._frame_count += 1
            else:
                consecutive_drops += 1
                if consecutive_drops == 1:
                    logger.warning("Frame drop detected.")
                if consecutive_drops >= max_drops:
                    logger.error(
                        "Camera lost (%d consecutive drops). Attempting restart...",
                        consecutive_drops,
                    )
                    self._restart_camera()
                    consecutive_drops = 0

    # ------------------------------------------------------------------

    def read_bgr(self) -> tuple[bool, np.ndarray | None]:
        """Return the latest raw BGR frame (copy)."""
        with self._frame_lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the latest frame pre-processed: flipped, resized, RGB.

        Note: This calls ``read_bgr()`` internally so both methods always
        return frames from the same underlying snapshot.
        """
        ok, frame = self.read_bgr()
        if not ok or frame is None:
            return False, None
        return True, self._preprocess(frame)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Flip + optional resize + BGR→RGB."""
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------

    def _restart_camera(self) -> None:
        """Attempt to reopen the camera after a disconnect."""
        with self._cap_lock:
            if self._cap is not None:
                with contextlib.suppress(Exception):
                    self._cap.release()
                self._cap = None

        time.sleep(1.0)
        for attempt in range(3):
            cap = self._open_cap(self.camera_index)
            if cap is not None:
                with self._cap_lock:
                    self._cap = cap
                logger.info("Camera reopened successfully (attempt %d).", attempt + 1)
                return
            time.sleep(2.0)
        logger.error("Failed to reopen camera after 3 attempts.")

    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        logger.info("FrameGrabber stopped.")

    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Total frames captured since start."""
        return self._frame_count

    @property
    def actual_resolution(self) -> tuple[int, int]:
        """Last known camera resolution (updated by capture loop, thread-safe)."""
        return self._actual_w, self._actual_h

    # ------------------------------------------------------------------

    def __enter__(self) -> FrameGrabber:
        """Start capture on context entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop capture on context exit."""
        self.stop()
