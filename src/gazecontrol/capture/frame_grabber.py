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
from typing import Any

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

        self._frame: np.ndarray[Any, Any] | None = None
        self._frame_lock = threading.Lock()
        self._frame_count: int = 0

        # Cached resolution snapshot — updated by capture loop, never by consumer.
        self._actual_w: int = 0
        self._actual_h: int = 0

        self._running = threading.Event()  # set() = running, clear() = stop requested
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
                    self.width,
                    self.height,
                    actual_w,
                    actual_h,
                )
                self.width = actual_w
                self.height = actual_h
            self._actual_w = actual_w
            self._actual_h = actual_h

            # Warm-up: Windows/DSHOW cameras return black frames (mean≈0)
            # for up to ~500 ms after cap.set() calls while the driver
            # re-initialises.  Read with a 2-second timeout until we see
            # a non-black frame.  warmup_frames is the maximum read count
            # (safety cap); the loop exits early on the first real frame.
            s = get_settings().camera
            n_warmup = max(20, s.warmup_frames)
            deadline = time.monotonic() + 2.0
            last_frame: np.ndarray[Any, Any] | None = None
            for _ in range(n_warmup):
                ret, frame = cap.read()
                if not ret:
                    logger.error("Camera opened but warmup read failed.")
                    cap.release()
                    return False
                last_frame = frame
                if frame is not None and frame.mean() > 1.0:
                    break  # got a real frame
                if time.monotonic() >= deadline:
                    break  # timeout — use whatever we have

            if last_frame is None:
                cap.release()
                return False

            logger.debug("Camera warm-up done (mean=%.1f).", last_frame.mean())

            self._cap = cap

        with self._frame_lock:
            self._frame = last_frame

        self._running.set()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("FrameGrabber started: %dx%d@%dfps", self.width, self.height, self.fps)
        return True

    def _open_cap(self, index: int | None) -> cv2.VideoCapture | None:
        """Open VideoCapture, preferring MSMF on Windows over DirectShow.

        MSMF (Media Foundation) correctly handles resolution changes via
        cap.set() and produces real frames immediately after initialisation.
        DSHOW often returns black frames permanently after cap.set() on
        Windows 10/11 — it is tried last as a fallback only.

        ``CAP_PROP_AUTO_EXPOSURE`` is set to manual (0.25) only when the
        setting ``camera.auto_exposure == "manual"`` is requested.  Leaving
        it unset (auto) is the correct default — forcing manual exposure
        produces dark/black frames.
        """
        if index is None:
            logger.error("Camera index is None — cannot open capture device.")
            return None
        auto_exposure = get_settings().camera.auto_exposure

        # Backend priority: MSMF first on Windows (works correctly with
        # cap.set()), then generic CAP_ANY, then DSHOW as last resort.
        backends = [cv2.CAP_MSMF, cv2.CAP_ANY, cv2.CAP_DSHOW]

        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            if auto_exposure == "manual":
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                logger.debug("FrameGrabber: exposure MANUAL.")
            else:
                logger.debug("FrameGrabber: exposure AUTO.")
            backend_name = {
                cv2.CAP_MSMF: "MSMF",
                cv2.CAP_DSHOW: "DSHOW",
                cv2.CAP_ANY: "ANY",
            }.get(backend, str(backend))
            logger.info("FrameGrabber: opened camera %d via %s.", index, backend_name)
            return cap

        logger.error("Cannot open camera index=%d (tried MSMF, ANY, DSHOW).", index)
        return None

    # ------------------------------------------------------------------
    # Capture loop (background thread)
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Read frames continuously, refreshing the shared frame buffer."""
        consecutive_drops = 0
        max_drops = 30  # ~1 s at 30 fps before restart attempt

        while self._running.is_set():
            # Snapshot the cap pointer under the lock, then release the lock
            # BEFORE calling cap.read() — a stalled camera (e.g. unplugged
            # webcam) would otherwise block stop() / restart for seconds.
            # _restart_camera() never frees the underlying ``cv2.VideoCapture``
            # while a read is in flight: it sets ``self._cap = None`` first,
            # so a subsequent read on the captured local pointer returns
            # ``ret=False`` (cv2 sentinel) which is already handled below.
            with self._cap_lock:
                cap = self._cap

            if cap is None:
                time.sleep(0.033)
                continue
            try:
                ret, frame = cap.read()
            except cv2.error as exc:
                logger.debug("cap.read() raised cv2.error: %s", exc)
                ret, frame = False, None

            if ret:
                consecutive_drops = 0
                with self._frame_lock:
                    self._frame = frame
                    self._frame_count += 1
            else:
                consecutive_drops += 1
                if consecutive_drops == 1:
                    logger.warning("Frame drop detected.")
                    # Re-cache resolution after drop (backend may have changed it).
                    with self._cap_lock:
                        if self._cap is not None:
                            self._actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            self._actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if consecutive_drops >= max_drops:
                    logger.error(
                        "Camera lost (%d consecutive drops). Attempting restart...",
                        consecutive_drops,
                    )
                    self._restart_camera()
                    consecutive_drops = 0

    # ------------------------------------------------------------------

    def read_bgr(self) -> tuple[bool, np.ndarray[Any, Any] | None]:
        """Return the latest raw BGR frame.

        Returns the producer's frame buffer as a read-only ndarray view —
        no copy.  Consumers MUST treat the returned array as immutable and
        copy before mutating.  Saves ~10–30 % GC pressure at 30 fps on
        constrained hardware.
        """
        with self._frame_lock:
            if self._frame is None:
                return False, None
            view = self._frame.view()
            view.flags.writeable = False
            return True, view

    # ------------------------------------------------------------------

    def _restart_camera(self) -> None:
        """Attempt to reopen the camera after a disconnect.

        The old camera handle is released immediately, then re-open attempts
        are made **without** holding ``_cap_lock`` during sleep — so ``stop()``
        and ``actual_resolution`` are never blocked for the full retry window.
        """
        with self._cap_lock:
            if self._cap is not None:
                with contextlib.suppress(Exception):
                    self._cap.release()
                self._cap = None

        # Sleep and retry loop runs outside the lock to avoid blocking consumers.
        time.sleep(1.0)
        for attempt in range(3):
            if not self._running.is_set():
                return  # stop() was called while we were sleeping
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
        """Stop the capture thread and release the camera.  Idempotent."""
        if not self._running.is_set():
            return  # already stopped or never started
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
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
