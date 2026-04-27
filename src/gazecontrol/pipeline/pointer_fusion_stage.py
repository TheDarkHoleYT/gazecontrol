"""PointerFusionStage — combine hand fingertip and gaze into a single pointer.

Priority (top to bottom):

    1. Hand is tracked with confidence ≥ ``hand_confidence_threshold``
       → use ``ctx.fingertip_screen``.
    2. Gaze is in a fixation with confidence ≥ ``gaze_confidence_threshold``
       → use the fixation centroid.
    3. Gaze is otherwise valid (saccade / pursuit) and confidence ≥ threshold
       → use the raw gaze point.
    4. None of the above → ``pointer_screen = None``.

Conflict resolution: when both hand and gaze are valid but diverge by more
than ``divergence_threshold_px``, hand wins (the user is explicitly pointing).

Optional ``gaze_assisted_click``: when the hand is idle (no recent motion)
and a strong fixation is available, the fixation centroid overrides the
fingertip. Off by default.
"""

from __future__ import annotations

import logging
import math

from gazecontrol.pipeline.context import FrameContext
from gazecontrol.settings import AppSettings, FusionSettings, get_settings

logger = logging.getLogger(__name__)


class PointerFusionStage:
    """Decides which spatial source drives :attr:`FrameContext.pointer_screen`."""

    name = "pointer_fusion"

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings

    def start(self) -> bool:
        """No resources to allocate."""
        return True

    def stop(self) -> None:
        """No resources to release."""

    def _cfg(self) -> FusionSettings:
        if self._settings is not None:
            return self._settings.fusion
        return get_settings().fusion

    def process(self, ctx: FrameContext) -> FrameContext:
        """Populate ``ctx.pointer_screen`` and ``ctx.pointer_source`` for the tick."""
        cfg = self._cfg()

        hand_xy = ctx.fingertip_screen
        gaze_xy = ctx.gaze_screen
        gaze_conf = ctx.gaze_confidence
        hand_ok = hand_xy is not None and ctx.gesture_confidence >= cfg.hand_confidence_threshold
        gaze_ok = gaze_xy is not None and gaze_conf >= cfg.gaze_confidence_threshold
        fixation_centroid = self._fixation_centroid(ctx)

        # 1. Hand priority — but allow gaze-assisted click when hand idle.
        if hand_ok and hand_xy is not None:
            if (
                cfg.gaze_assisted_click
                and fixation_centroid is not None
                and self._hand_is_idle(ctx)
            ):
                ctx.pointer_screen = fixation_centroid
                ctx.pointer_source = "fused"
                return ctx
            if gaze_ok and gaze_xy is not None:
                # Both valid; diverge check is informational only — hand wins.
                divergence = math.hypot(hand_xy[0] - gaze_xy[0], hand_xy[1] - gaze_xy[1])
                if divergence > cfg.divergence_threshold_px:
                    logger.debug(
                        "PointerFusion: hand/gaze divergence=%.0fpx — hand wins.",
                        divergence,
                    )
            ctx.pointer_screen = hand_xy
            ctx.pointer_source = "hand"
            return ctx

        # 2. Gaze fixation centroid (most stable).
        if fixation_centroid is not None:
            ctx.pointer_screen = fixation_centroid
            ctx.pointer_source = "gaze"
            return ctx

        # 3. Raw gaze (saccade / pursuit / no-fixation).
        if gaze_ok and gaze_xy is not None:
            ctx.pointer_screen = gaze_xy
            ctx.pointer_source = "gaze"
            return ctx

        # 4. Fallback to whatever hand we may have, even with low confidence.
        if hand_xy is not None:
            ctx.pointer_screen = hand_xy
            ctx.pointer_source = "hand"
            return ctx

        ctx.pointer_screen = None
        ctx.pointer_source = "none"
        return ctx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fixation_centroid(self, ctx: FrameContext) -> tuple[int, int] | None:
        ev = ctx.gaze_event
        if ev is None or ev.type != "fixation" or ev.centroid is None:
            return None
        if ctx.gaze_confidence < self._cfg().gaze_confidence_threshold:
            return None
        return (int(ev.centroid[0]), int(ev.centroid[1]))

    def _hand_is_idle(self, ctx: FrameContext) -> bool:
        """Heuristic: hand idle when gesture confidence is high but no scroll/pinch."""
        return (
            ctx.pinch_event is None
            and ctx.two_finger_scroll_delta == 0
            and ctx.gesture_confidence < 0.4
        )
