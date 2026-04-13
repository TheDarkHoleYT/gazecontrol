"""IntentStage — window selection and intent FSM."""
from __future__ import annotations

import logging

from gazecontrol.intent.state_machine import IntentStateMachine
from gazecontrol.intent.window_selector import WindowSelector
from gazecontrol.pipeline.context import FrameContext

logger = logging.getLogger(__name__)


class IntentStage:
    """Maps gaze + gesture to a window action via the FSM."""

    def __init__(self) -> None:
        self._state_machine = IntentStateMachine()
        self._window_selector = WindowSelector()

    @property
    def state(self) -> str:
        return self._state_machine.state

    def process(self, ctx: FrameContext) -> FrameContext:
        """Run window selection + FSM update."""
        target_window = None
        if ctx.gaze_point:
            target_window = self._window_selector.find_window(ctx.gaze_point)
        ctx.target_window = target_window

        action_dict = self._state_machine.update(
            gaze_point=ctx.gaze_point,
            target_window=target_window,
            gesture_id=ctx.gesture_label,
            gesture_confidence=ctx.gesture_confidence,
            hand_position=ctx.hand_position,
        )
        # Store raw dict for now; typed Action wrapping happens in orchestrator.
        ctx.action = action_dict  # type: ignore[assignment]
        return ctx
