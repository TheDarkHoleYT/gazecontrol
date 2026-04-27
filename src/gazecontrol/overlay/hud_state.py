"""HudState — typed snapshot of HUD display data for hand-only control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HudState:
    """Data required to render one HUD frame.

    Immutable snapshot — callers build a fresh ``HudState`` per tick and
    swap the reference atomically (Python's GIL makes single-attribute
    pointer swaps thread-safe). Frozen prevents in-place mutation that
    would cause cross-thread races between the pipeline producer and the
    Qt renderer.
    """

    fingertip_screen: tuple[int, int] | None = None
    state: str = "IDLE"
    hovered_window: Any = None  # HoveredWindow | None
    gesture_id: str | None = None
    gesture_confidence: float = 0.0
    interaction_kind: str | None = None  # InteractionKind value or None
    launcher_visible: bool = False
    # Eye-tracking enrichment (None when input_mode == HAND_ONLY).
    gaze_screen: tuple[int, int] | None = None
    gaze_confidence: float = 0.0
    pointer_source: str = "hand"
    input_mode: str = "hand"
