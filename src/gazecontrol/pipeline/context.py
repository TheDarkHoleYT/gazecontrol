"""FrameContext — typed container passed through pipeline stages.

Each stage reads the context, processes its portion, and returns an updated
copy (or mutates in place — stages own their output fields).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from gazecontrol.capture.frame_preprocessor import FrameQuality
    from gazecontrol.gaze.fixation_detector import GazeEvent
    from gazecontrol.intent.types import Action


@dataclass
class FrameContext:
    """Mutable accumulator for one pipeline tick.

    Stages add their outputs; downstream stages read them.
    """

    # --- CaptureStage outputs ---
    frame_bgr: "np.ndarray | None" = None
    frame_rgb: "np.ndarray | None" = None
    quality: "FrameQuality | None" = None
    capture_ok: bool = False

    # --- GazeStage outputs ---
    landmarks: Any = None                              # mediapipe result / eyetrax features
    gaze_raw: tuple[float, float] | None = None        # TinyMLP screen coords
    gaze_l2cs: tuple[float, float] | None = None       # L2CS screen coords (optional)
    gaze_filtered: tuple[float, float] | None = None   # after 1€ filter
    gaze_corrected: tuple[float, float] | None = None  # after drift correction
    gaze_point: tuple[int, int] | None = None          # final integer pixel
    fixation_event: "GazeEvent | None" = None
    blink: bool = False

    # --- GestureStage outputs ---
    hand_result: Any = None
    gesture_label: str | None = None
    gesture_confidence: float = 0.0
    hand_position: tuple[float, float] | None = None   # screen coords

    # --- IntentStage outputs ---
    target_window: Any = None  # HWND or window info dict
    action: "Action | None" = None

    # --- Timing ---
    t0: float = field(default=0.0)                     # monotonic timestamp at tick start
