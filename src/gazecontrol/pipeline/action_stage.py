"""ActionStage — dispatches interaction events to the window manager.

This is the terminal stage of the hand-only pipeline.  It reads the typed
:class:`~gazecontrol.interaction.types.Interaction` produced by
``InteractionStage`` and routes it to the appropriate platform call:

- ``CLICK``         → synthetic left click at fingertip position.
- ``DRAG_UPDATE``   → ``WindowsManager.move_window``.
- ``RESIZE_UPDATE`` → ``WindowsManager.resize_window``.
- ``SCROLL_UP/DOWN``→ ``WindowsManager.scroll_at``.
- ``TOGGLE_LAUNCHER`` → ``overlay_bridge.toggle_launcher()``.

Shutdown: an external ``on_stop`` callback is supported so the pipeline can
be halted cleanly from a gesture (e.g. a future "exit" gesture).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from gazecontrol.interaction.types import Interaction, InteractionKind
from gazecontrol.pipeline.context import FrameContext

logger = logging.getLogger(__name__)


class ActionStage:
    """Pipeline stage: execute interactions and dispatch to the window manager.

    Args:
        window_manager: Platform window manager (``BaseWindowManager``).
        app_launcher:   Optional :class:`~gazecontrol.window_manager.launcher.AppLauncher`.
        overlay_bridge: Optional callable ``() -> None`` invoked for TOGGLE_LAUNCHER.
        on_stop:        Optional callback invoked to shut down the pipeline.
    """

    name = "action"

    def __init__(
        self,
        window_manager: Any,
        app_launcher: Any | None = None,
        overlay_bridge: Any | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self._wm = window_manager
        self._launcher = app_launcher
        self._overlay_bridge = overlay_bridge
        self._on_stop = on_stop

    def start(self) -> bool:
        """No resources to open."""
        return True

    def stop(self) -> None:
        """No resources to release."""

    def process(self, ctx: FrameContext) -> FrameContext:
        """Execute the interaction stored in ``ctx.interaction``, if any."""
        interaction: Interaction | None = ctx.interaction
        if interaction is None:
            return ctx

        kind = interaction.kind
        window = interaction.window
        point = interaction.point
        data = interaction.data

        try:
            if kind == InteractionKind.CLICK:
                self._wm.click_at(*point)

            elif kind == InteractionKind.TOGGLE_LAUNCHER:
                if self._overlay_bridge is not None:
                    self._overlay_bridge()
                else:
                    logger.debug("ActionStage: TOGGLE_LAUNCHER — no overlay_bridge wired.")

            elif kind == InteractionKind.DRAG_UPDATE:
                if window is not None:
                    self._wm.move_window(window.hwnd, data["new_x"], data["new_y"])

            elif kind == InteractionKind.DRAG_END:
                pass  # drag ended; window is already in its final position

            elif kind == InteractionKind.RESIZE_UPDATE:
                if window is not None:
                    self._wm.resize_window(
                        window.hwnd,
                        data["new_x"],
                        data["new_y"],
                        data["new_w"],
                        data["new_h"],
                    )

            elif kind == InteractionKind.RESIZE_END:
                pass  # resize ended

            elif kind in (InteractionKind.SCROLL_UP, InteractionKind.SCROLL_DOWN):
                delta = 120 if kind == InteractionKind.SCROLL_UP else -120
                self._wm.scroll_at(*point, delta=delta)

            elif kind in (InteractionKind.DRAG_START, InteractionKind.RESIZE_START):
                # Start events are informational — no WM call needed.
                pass

        except Exception:
            logger.warning("ActionStage: dispatch failed for %s", kind, exc_info=True)

        return ctx
