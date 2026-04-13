"""Shared intent types — Enums and Action dataclass.

All cross-module communication about gestures, states and window actions goes
through these typed containers instead of raw strings/dicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IntentState(StrEnum):
    """FSM states for the intent state machine."""

    IDLE = "IDLE"
    TARGETING = "TARGETING"
    READY = "READY"
    DRAG = "DRAG"
    RESIZE = "RESIZE"
    COOLDOWN = "COOLDOWN"


class GestureLabel(StrEnum):
    """Known gesture labels produced by classifiers."""

    GRAB = "GRAB"
    RELEASE = "RELEASE"
    PINCH = "PINCH"
    SWIPE_LEFT = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"
    CLOSE_SIGN = "CLOSE_SIGN"
    SCROLL_UP = "SCROLL_UP"
    SCROLL_DOWN = "SCROLL_DOWN"
    MAXIMIZE = "MAXIMIZE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, s: str | None) -> GestureLabel:
        """Parse a string produced by a classifier, defaulting to UNKNOWN."""
        if s is None:
            return cls.UNKNOWN
        try:
            return cls(s)
        except ValueError:
            return cls.UNKNOWN


class ActionType(StrEnum):
    """Window action types dispatched to WindowManager."""

    DRAG = "DRAG"
    RESIZE = "RESIZE"
    CLOSE = "CLOSE"
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"
    BRING_FRONT = "BRING_FRONT"
    SCROLL_UP = "SCROLL_UP"
    SCROLL_DOWN = "SCROLL_DOWN"
    CLOSE_APP = "CLOSE_APP"  # special: shut down GazeControl itself


class DragPhase(StrEnum):
    """Sub-phase for DRAG and RESIZE actions."""

    START = "start"
    MOVE = "move"
    END = "end"


@dataclass(frozen=True)
class Action:
    """A typed window action emitted by the IntentStateMachine.

    Attributes:
        type: What kind of action to perform.
        window: Target window info dict with at least ``hwnd`` key.
        data: Action-specific payload (delta, start_rect, phase, etc.).
    """

    type: ActionType
    window: dict  # {'hwnd': int, 'rect': (x,y,w,h), ...}
    data: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Ensure data defaults to an empty dict when None is passed."""
        object.__setattr__(self, "data", self.data or {})

    def to_legacy_dict(self) -> dict:
        """Convert to the legacy action dict format for backward compatibility."""
        return {
            "type": self.type.value,
            "window": self.window,
            "data": self.data,
        }
