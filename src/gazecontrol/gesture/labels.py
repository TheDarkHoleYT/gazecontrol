"""GestureLabel — canonical gesture label vocabulary.

Single source of truth for gesture names shared by:

- :class:`~gazecontrol.gesture.rule_classifier.RuleClassifier`
- :class:`~gazecontrol.gesture.mlp_classifier.MLPClassifier`
- :class:`~gazecontrol.interaction.interaction_fsm.InteractionFSM`
- :class:`~gazecontrol.settings.GestureLabelsSettings`

Using :class:`GestureLabel` instead of raw strings eliminates the label
mismatch between the rule-based layer (which previously emitted ``CLOSE_SIGN``,
``GRAB``, etc.) and the MLP layer (which emitted ``MAXIMIZE``, ``SWIPE_LEFT``).

Migration
---------
Code that compared against bare string literals should be updated to compare
against ``GestureLabel`` members or their ``.value`` strings::

    # Before
    if gesture == "PINCH":
        ...

    # After (preferred)
    if gesture == GestureLabel.PINCH:
        ...

    # Also valid (backward-compat with string comparisons)
    if gesture == GestureLabel.PINCH.value:
        ...
"""

from __future__ import annotations

from enum import StrEnum


class GestureLabel(StrEnum):
    """Canonical gesture label vocabulary.

    Because :class:`GestureLabel` extends :class:`~enum.StrEnum`, instances
    compare equal to their ``value`` string::

        GestureLabel.PINCH == "PINCH"  # True

    This means existing string comparisons in the FSM and HUD work without
    modification while new code can use the enum directly.
    """

    PINCH = "PINCH"
    """Index fingertip touches thumb — primary action trigger."""

    RELEASE = "RELEASE"
    """Hand open after a pinch."""

    SCROLL_UP = "SCROLL_UP"
    """Index + middle fingers extended, moving upward."""

    SCROLL_DOWN = "SCROLL_DOWN"
    """Index + middle fingers extended, moving downward."""

    SWIPE_LEFT = "SWIPE_LEFT"
    """Fast horizontal sweep to the left."""

    SWIPE_RIGHT = "SWIPE_RIGHT"
    """Fast horizontal sweep to the right."""

    GRAB = "GRAB"
    """All fingers curled (fist-like)."""

    CLOSE_SIGN = "CLOSE_SIGN"
    """Thumb pointing down — map to close/quit action."""

    MAXIMIZE = "MAXIMIZE"
    """All fingers extended, palm facing camera."""


# Ordered list used as the default label vocabulary for classifier training
# and the :class:`~gazecontrol.settings.GestureLabelsSettings` default.
DEFAULT_LABELS: list[str] = [label.value for label in GestureLabel]
