"""InteractionStage — drives the InteractionFSM and resolves hovered window."""

from __future__ import annotations

import logging

from gazecontrol.gesture.pinch_tracker import PinchEvent
from gazecontrol.interaction.interaction_fsm import InteractionFSM
from gazecontrol.interaction.types import Interaction
from gazecontrol.interaction.window_hit_tester import WindowHitTester
from gazecontrol.pipeline.context import FrameContext

logger = logging.getLogger(__name__)


class InteractionStage:
    """Maps per-frame gesture context to typed :class:`Interaction` events.

    Reads:  ``ctx.fingertip_screen``, ``ctx.pinch_event``, ``ctx.two_finger_scroll_delta``
    Writes: ``ctx.hovered_window``, ``ctx.interaction``
    """

    name = "interaction"

    def __init__(
        self,
        fsm: InteractionFSM | None = None,
        hit_tester: WindowHitTester | None = None,
    ) -> None:
        self._fsm = fsm or InteractionFSM()
        self._hit_tester = hit_tester or WindowHitTester()

    @property
    def state(self) -> str:
        """Current FSM state name (useful for HUD rendering)."""
        return self._fsm.state

    def start(self) -> bool:
        """No-op: resources are allocated in ``__init__``."""
        return True

    def stop(self) -> None:
        """No-op: nothing to release."""

    def process(self, ctx: FrameContext) -> FrameContext:
        """Run hit-test + FSM update.

        Uses ``ctx.pointer_screen`` when present (set by GestureStage in
        HAND_ONLY mode, by PointerFusionStage in EYE_HAND mode); falls
        back to ``ctx.fingertip_screen`` so older callers stay valid.
        """
        pointer = ctx.pointer_screen if ctx.pointer_screen is not None else ctx.fingertip_screen

        # Hovered window — always resolved for HUD rendering.
        if pointer is not None:
            ctx.hovered_window = self._hit_tester.at(pointer)
        else:
            ctx.hovered_window = None

        pinch_event = ctx.pinch_event if ctx.pinch_event is not None else PinchEvent.NONE

        try:
            interaction: Interaction | None = self._fsm.update(
                pinch_event=pinch_event,
                fingertip_screen=pointer,
                hit_tester=self._hit_tester,
                two_finger_scroll_delta=ctx.two_finger_scroll_delta,
            )
        except Exception:
            logger.warning("InteractionStage: FSM update failed", exc_info=True)
            interaction = None

        ctx.interaction = interaction
        return ctx
