"""CaptureStage — webcam capture + frame preprocessing."""

from __future__ import annotations

import logging

import cv2

from gazecontrol.capture.frame_grabber import FrameGrabber
from gazecontrol.capture.frame_preprocessor import FramePreprocessor
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.settings import AppSettings, get_settings

logger = logging.getLogger(__name__)


class CaptureStage:
    """Wraps FrameGrabber + FramePreprocessor into a single pipeline stage.

    ``process()`` grabs one frame (BGR), derives the RGB copy from it atomically,
    runs CLAHE preprocessing, and stores everything on the context.
    """

    name = "capture"
    # Signals to PipelineEngine that downstream stages should be skipped when
    # this stage sets capture_ok=False (no usable frame this tick).
    skip_on_capture_fail = True

    def __init__(self, settings: AppSettings | None = None) -> None:
        s = (settings or get_settings()).camera
        self.grabber = FrameGrabber(
            camera_index=s.index,
            width=s.width,
            height=s.height,
            fps=s.fps,
        )
        self._enhance = s.enhance
        self._preprocessor = FramePreprocessor(blur_threshold=s.blur_threshold)

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

        # Mirror horizontally so the frame matches the user's natural perspective
        # (webcam frames are un-mirrored by default; mirror = natural feedback).
        # Both BGR and RGB are derived from the same flipped snapshot so that
        # gaze and gesture coordinate systems are consistent.
        frame_bgr_flipped = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr_flipped, cv2.COLOR_BGR2RGB)

        if self._enhance:
            # CLAHE + sharpening: helps in poor lighting but costs ~3–5 ms/frame.
            enhanced_bgr, quality = self._preprocessor.process(frame_bgr_flipped)
        else:
            # Skip CLAHE/sharpening; compute quality score only (cheap Laplacian).
            quality = self._preprocessor.compute_quality(frame_bgr_flipped)
            enhanced_bgr = frame_bgr_flipped

        ctx.capture_ok = True
        ctx.frame_bgr = enhanced_bgr  # flipped (CLAHE-enhanced when enhance=True)
        ctx.frame_rgb = frame_rgb  # flipped, raw RGB (for hand detector)
        ctx.quality = quality
        return ctx

    def stop(self) -> None:
        """Stop the grabber thread and release the camera."""
        self.grabber.stop()
