"""PipelineFactory — assemble a :class:`PipelineEngine` for a given input mode.

Each mode produces a different stage list:

- ``InputMode.HAND_ONLY``: ``[Capture, Gesture, Interaction, Action]``
  (identical to the pre-feature behaviour — no regression).
- ``InputMode.EYE_HAND``:  ``[Capture, Gaze, Gesture, PointerFusion, Interaction, Action]``.

The factory owns the cross-stage wiring (``on_stop`` callback, overlay
bridge, drift feedback from ActionStage to GazeStage) so that ``cli.py``
just calls :func:`build` and gets a fully configured engine + thread.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from gazecontrol.errors import GazeControlError
from gazecontrol.gaze.backend import GazeBackend
from gazecontrol.gesture.fingertip_mapper import FingertipMapper, VirtualDesktop
from gazecontrol.interaction.interaction_fsm import InteractionFSM
from gazecontrol.interaction.window_hit_tester import WindowHitTester
from gazecontrol.pipeline.action_stage import ActionStage
from gazecontrol.pipeline.capture_stage import CaptureStage
from gazecontrol.pipeline.engine import PipelineEngine
from gazecontrol.pipeline.gaze_stage import GazeStage
from gazecontrol.pipeline.gesture_stage import GestureStage
from gazecontrol.pipeline.interaction_stage import InteractionStage
from gazecontrol.pipeline.stage import PipelineStage
from gazecontrol.runtime.input_mode import InputMode
from gazecontrol.settings import AppSettings, get_settings
from gazecontrol.window_manager.launcher import AppLauncher, LauncherApp
from gazecontrol.window_manager.windows_mgr import WindowsManager

logger = logging.getLogger(__name__)


@dataclass
class BuiltPipeline:
    """Container returned by :func:`PipelineFactory.build`."""

    engine: PipelineEngine
    interaction_stage: InteractionStage
    gaze_stage: GazeStage | None
    launcher_apps: list[LauncherApp]
    app_launcher: AppLauncher
    overlay_bridge_holder: list[Callable[[], None]]


class PipelineFactory:
    """Builds the per-mode stage pipeline."""

    def __init__(
        self,
        mode: InputMode,
        vdesk: tuple[int, int, int, int],
        settings: AppSettings | None = None,
    ) -> None:
        self._mode = mode
        self._vdesk = vdesk
        self._settings = settings or get_settings()

    def build(self) -> BuiltPipeline:
        """Construct stages, engine, and overlay-bridge plumbing for the mode."""
        s = self._settings
        left, top, width, height = self._vdesk
        screen_w = width
        screen_h = height

        desktop = VirtualDesktop(left=left, top=top, width=width, height=height)
        mapper = FingertipMapper(
            desktop,
            sensitivity=s.gesture.pointer_sensitivity,
            filter_cfg=s.gesture.pointer,
        )

        icfg = s.interaction
        fsm = InteractionFSM(
            tap_ms=icfg.tap_ms,
            hold_ms=icfg.hold_ms,
            double_ms=icfg.double_pinch_ms,
            tap_max_move_px=icfg.tap_max_move_px,
            grip_ratio=icfg.grip_ratio,
            cooldown_ms=icfg.cooldown_ms,
            min_w=icfg.min_window_width,
            min_h=icfg.min_window_height,
        )
        hit_tester = WindowHitTester()

        capture_stage = CaptureStage(settings=s)
        gesture_stage = GestureStage(fingertip_mapper=mapper, settings=s)
        interaction_stage = InteractionStage(fsm=fsm, hit_tester=hit_tester)
        window_manager = WindowsManager()

        gaze_stage = None
        pointer_fusion_stage = None
        if self._mode == InputMode.EYE_HAND:
            gaze_stage = self._build_gaze_stage(screen_w, screen_h)
            from gazecontrol.pipeline.pointer_fusion_stage import PointerFusionStage

            pointer_fusion_stage = PointerFusionStage(settings=s)

        # Build app launcher.
        app_launcher = AppLauncher()
        launcher_apps: list[LauncherApp] = [
            LauncherApp(name=a.name, exe=a.exe, args=tuple(a.args), icon=a.icon)
            for a in s.launcher.apps
        ]

        engine_holder: list[PipelineEngine] = []
        overlay_bridge_holder: list[Callable[[], None]] = []

        def _request_stop() -> None:
            if engine_holder:
                engine_holder[0].request_stop()

        def _toggle_launcher() -> None:
            if overlay_bridge_holder:
                overlay_bridge_holder[0]()

        action_stage = ActionStage(
            window_manager=window_manager,
            app_launcher=app_launcher,
            overlay_bridge=_toggle_launcher,
            on_stop=_request_stop,
        )

        # Order matters: gaze before gesture so fusion sees both per tick.
        stages: list[PipelineStage] = [capture_stage]
        if gaze_stage is not None:
            stages.append(gaze_stage)
        stages.append(gesture_stage)
        if pointer_fusion_stage is not None:
            stages.append(pointer_fusion_stage)
        stages.extend([interaction_stage, action_stage])

        engine = PipelineEngine(stages=stages, settings=s)
        engine_holder.append(engine)

        logger.info(
            "PipelineFactory: built %d stages for mode=%s",
            len(stages),
            self._mode.value,
        )
        return BuiltPipeline(
            engine=engine,
            interaction_stage=interaction_stage,
            gaze_stage=gaze_stage,
            launcher_apps=launcher_apps,
            app_launcher=app_launcher,
            overlay_bridge_holder=overlay_bridge_holder,
        )

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _build_gaze_stage(self, screen_w: int, screen_h: int) -> GazeStage:
        backend = self._build_gaze_backend(screen_w, screen_h)
        return GazeStage(
            backend=backend,
            screen_w=screen_w,
            screen_h=screen_h,
            settings=self._settings,
        )

    def _build_gaze_backend(self, screen_w: int, screen_h: int) -> GazeBackend:
        gcfg = self._settings.gaze
        choice = gcfg.backend
        if choice == "eyetrax":
            from gazecontrol.gaze.eyetrax_backend import EyetraxBackend

            return EyetraxBackend(screen_w, screen_h, profile_name=gcfg.profile)
        if choice == "l2cs":
            from gazecontrol.gaze.l2cs_backend import L2CSBackend

            return L2CSBackend(
                screen_w,
                screen_h,
                profile_name=gcfg.profile,
                strict=gcfg.strict_l2cs,
            )
        if choice == "ensemble":
            from gazecontrol.gaze.ensemble_backend import EnsembleBackend
            from gazecontrol.gaze.eyetrax_backend import EyetraxBackend
            from gazecontrol.gaze.l2cs_backend import L2CSBackend

            primary = EyetraxBackend(screen_w, screen_h, profile_name=gcfg.profile)
            secondary = L2CSBackend(
                screen_w,
                screen_h,
                profile_name=gcfg.profile,
                strict=False,
            )
            return EnsembleBackend(
                primary=primary,
                secondary=secondary,
                weight_primary=gcfg.ensemble_weight_eyetrax,
                weight_secondary=gcfg.ensemble_weight_l2cs,
            )
        raise GazeControlError(f"Unknown gaze backend: {choice!r}")
