"""Interaction package — hand-based desktop control logic.

Replaces the gaze-based intent package with a purely hand-driven model:

- :class:`~gazecontrol.interaction.types.HoveredWindow` — typed window info.
- :class:`~gazecontrol.interaction.types.Interaction` — typed action value object.
- :class:`~gazecontrol.interaction.interaction_fsm.InteractionFSM` — pinch → action FSM.
- :class:`~gazecontrol.interaction.window_hit_tester.WindowHitTester` — point-in-window lookup.
- :func:`~gazecontrol.interaction.grip_region.is_in_resize_grip` — resize corner detection.
"""
