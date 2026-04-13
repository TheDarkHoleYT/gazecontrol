"""CaptureStage — webcam capture + frame preprocessing.

Fixes vs the old main.py approach:
- Single frame read per tick: BGR and RGB derived from one copy (no double copy bug).
- Busy-loop guard: returns immediately with capture_ok=False when no frame available.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from gazecontrol.capture.frame_grabber import FrameGrabber
from gazecontrol.capture.frame_preprocessor import FramePreprocessor
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.settings import get_settings

logger = logging.getLogger(__name__)


class CaptureStage:
    """Wraps FrameGrabber + FramePreprocessor into a single pipeline stage.

    ``process()`` grabs one frame (BGR), derives the RGB copy from it atomically,
    runs CLAHE preprocessing, and stores everything on the context.
    """

    def __init__(self) -> None:
        s = get_settings().camera
        self.grabber = FrameGrabber(
            camera_index=s.index,
            width=s.width,
            height=s.height,
            fps=s.fps,
        )
        self._preprocessor = FramePreprocessor()

    def start(self) -> bool:
        """Start the background capture thread.  Returns False on camera failure."""
        return self.grabber.start()

    def process(self, ctx: FrameContext) -> FrameContext:
        """Read one frame and populate ctx.frame_bgr / frame_rgb / quality.

        If no frame is available (camera drop), sets ctx.capture_ok = False and
        returns immediately — the caller should skip this tick rather than busy-loop.
        """
        ok, frame_bgr = self.grabber.read_bgr()
        if not ok or frame_bgr is None:
            ctx.capture_ok = False
            return ctx

        # Derive RGB from the same raw frame (atomic — no second grabber.read() call).
        frame_rgb = cv2.cvtColor(cv2.flip(frame_bgr, 1), cv2.COLOR_BGR2RGB)

        # CLAHE enhancement runs on BGR; quality score computed inside.
        enhanced_bgr, quality = self._preprocessor.process(frame_bgr)

        ctx.capture_ok = True
        ctx.frame_bgr = enhanced_bgr
        ctx.frame_rgb = frame_rgb
        ctx.quality = quality
        return ctx

    def stop(self) -> None:
        """Stop the grabber thread and release the camera."""
        self.grabber.stop()
