"""Interaction types — value objects shared across the interaction package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class InteractionKind(StrEnum):
    """All possible interaction outcomes produced by :class:`InteractionFSM`."""

    CLICK = "CLICK"  # single tap → synthetic left click
    DOUBLE_CLICK = "DOUBLE_CLICK"  # two taps close together
    DRAG_START = "DRAG_START"  # pinch-hold began on a window
    DRAG_UPDATE = "DRAG_UPDATE"  # window is being dragged
    DRAG_END = "DRAG_END"  # drag released
    RESIZE_START = "RESIZE_START"  # pinch-hold in resize grip began
    RESIZE_UPDATE = "RESIZE_UPDATE"  # window is being resized
    RESIZE_END = "RESIZE_END"  # resize released
    SCROLL_UP = "SCROLL_UP"  # two-finger scroll up
    SCROLL_DOWN = "SCROLL_DOWN"  # two-finger scroll down
    TOGGLE_LAUNCHER = "TOGGLE_LAUNCHER"  # double-pinch → show/hide launcher


@dataclass(frozen=True)
class HoveredWindow:
    """Visible desktop window under the current fingertip position.

    Attributes:
        hwnd:  Win32 window handle.
        rect:  ``(x, y, width, height)`` in virtual-desktop pixels.
        title: Window title string.
    """

    hwnd: int
    rect: tuple[int, int, int, int]
    title: str = ""


@dataclass(frozen=True)
class Interaction:
    """A discrete interaction event produced by :class:`InteractionFSM`.

    Attributes:
        kind:   The type of interaction (see :class:`InteractionKind`).
        window: Target window, if any (``None`` for TOGGLE_LAUNCHER, CLICK on empty).
        point:  Screen coordinate ``(x, y)`` where the interaction is occurring.
        data:   Kind-specific payload (e.g. ``{"delta": (dx, dy)}``, ``{"phase": "start"}``).
    """

    kind: InteractionKind
    window: HoveredWindow | None = None
    point: tuple[int, int] = (0, 0)
    data: dict[str, Any] = field(default_factory=dict)
