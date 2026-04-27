"""QtPipelineThread — thin QThread adapter for PipelineEngine.

Wraps a PipelineEngine in a QThread and marshals per-frame callbacks to the
Qt event loop via pyqtSignal (QueuedConnection — thread-safe by default).

This module REQUIRES PyQt6.  Import only when Qt is available.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QThread, pyqtSignal

from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.engine import PipelineEngine

logger = logging.getLogger(__name__)


class QtPipelineThread(QThread):
    """Runs a ``PipelineEngine`` in a dedicated QThread.

    Emits ``frame_processed`` with each ``FrameContext`` after all stages
    complete for a tick.  The signal uses the default ``AutoConnection``,
    which becomes ``QueuedConnection`` when received across threads —
    ensuring the overlay / GUI slots are called on the main thread.

    Shutdown contract:
        ``stop()`` calls ``request_stop()`` on the engine *and* signals
        QThread interruption, then waits up to 3 s.  ``_emit_frame``
        skips emission once interruption is requested, so no signals
        fire after ``stop()`` returns — eliminating the race between
        ``isRunning()`` checks and signal delivery.
    """

    #: Emitted after every successfully captured tick.  Carries the FrameContext.
    frame_processed = pyqtSignal(object)

    def __init__(self, engine: PipelineEngine) -> None:
        super().__init__()
        self._engine = engine
        self._engine.set_on_frame(self._emit_frame)
        self._engine.set_on_shutdown(self._handle_shutdown)

    # ------------------------------------------------------------------

    def run(self) -> None:
        """QThread entry point — called by Qt when start() is invoked."""
        logger.info("QtPipelineThread: engine starting.")
        self._engine.run()

    def stop(self) -> None:
        """Request stop and wait up to 3 s for the thread to exit.

        If the thread does not exit within 3 s, it is forcibly terminated to
        prevent the process from hanging on shutdown.
        """
        self.requestInterruption()
        self._engine.request_stop()
        if not self.wait(3000):
            logger.warning("QtPipelineThread: did not exit within 3 s — terminating thread.")
            self.terminate()
            self.wait(1000)

    # ------------------------------------------------------------------

    def _emit_frame(self, ctx: FrameContext) -> None:
        """Called in the pipeline thread — emits signal to be received on GUI thread.

        Skips emission once interruption is requested so the GUI thread
        never receives stale frames after ``stop()`` returns.
        """
        if not self.isInterruptionRequested():
            self.frame_processed.emit(ctx)

    def _handle_shutdown(self) -> None:
        """Called by engine on clean exit — nothing extra needed here."""
