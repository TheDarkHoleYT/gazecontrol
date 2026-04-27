"""FrameContext — typed container passed through pipeline stages.

Each stage reads the context, mutates its own output fields in place, and
returns the same object.  The mutation-in-place convention is required by
the ``PipelineStage`` Protocol: every ``process()`` receives the *same*
``FrameContext`` accumulator and enriches it; downstream stages then read
the fields populated by upstream ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from gazecontrol.capture.frame_preprocessor import FrameQuality
    from gazecontrol.gaze.fixation_detector import GazeEvent
    from gazecontrol.gesture.pinch_tracker import PinchEvent
    from gazecontrol.interaction.types import HoveredWindow, Interaction


@dataclass
class FrameContext:
    """Mutable accumulator for one pipeline tick.

    Stages add their outputs by mutating fields; downstream stages read them.
    All fields default to ``None`` / ``False`` / ``0`` so a freshly
    constructed context is always safe to pass into the first stage.

    **Mutation convention**: every ``PipelineStage.process()`` receives
    *this* object and mutates only the fields it owns.  It must return the
    same object (not a copy).

    Field ownership:

    - ``CaptureStage``       → ``frame_rgb``, ``frame_bgr``, ``quality``, ``capture_ok``
    - ``GazeStage``          → ``gaze_screen``, ``gaze_confidence``, ``gaze_event``,
                               ``gaze_blink``, ``gaze_yaw_pitch_deg``, ``face_present``
    - ``GestureStage``       → ``hand_result``, ``features``, ``gesture_label``,
                               ``gesture_confidence``, ``fingertip_screen``,
                               ``pinch_event``, ``two_finger_scroll_delta``
    - ``PointerFusionStage`` → ``pointer_screen``, ``pointer_source``
    - ``InteractionStage``   → ``hovered_window``, ``interaction``
    - ``ActionStage``        → (reads ``interaction``; no new fields)
    """

    # --- CaptureStage outputs ---
    frame_bgr: np.ndarray[Any, Any] | None = None  # CLAHE-enhanced, flipped BGR
    frame_rgb: np.ndarray[Any, Any] | None = None  # flipped RGB — for hand detector
    quality: FrameQuality | None = None
    capture_ok: bool = False

    # --- GazeStage outputs (None when input_mode == HAND_ONLY) ---
    gaze_screen: tuple[int, int] | None = None
    gaze_confidence: float = 0.0
    gaze_event: GazeEvent | None = None
    gaze_blink: bool = False
    gaze_yaw_pitch_deg: tuple[float, float] | None = None
    face_present: bool = False

    # --- GestureStage outputs ---
    # hand_result: mediapipe HandLandmarkerResult or compatible (untyped upstream).
    hand_result: Any = None
    features: dict[str, Any] | None = None  # raw FeatureSet.to_dict()
    gesture_label: str | None = None
    gesture_confidence: float = 0.0
    fingertip_screen: tuple[int, int] | None = None
    pinch_event: PinchEvent | None = None
    two_finger_scroll_delta: int = 0

    # --- PointerFusionStage outputs (HAND_ONLY mode mirrors fingertip_screen) ---
    pointer_screen: tuple[int, int] | None = None
    pointer_source: str = "hand"

    # --- InteractionStage outputs ---
    hovered_window: HoveredWindow | None = None
    interaction: Interaction | None = None

    # --- Timing ---
    t0: float = field(default=0.0)
    frame_id: int = 0
