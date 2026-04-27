"""GestureFusion — confidence-weighted fusion of rule and ML classifiers.

Strategy
--------
1. **Rule classifier** (high precision, lower recall): if confident (≥ threshold),
   use its output directly.
2. **ML classifier** (higher recall, probabilistic): if the rule is uncertain and
   the ML score is confident (≥ threshold) *and* the label is in the FSM
   vocabulary, use the ML output.
3. **Rule passthrough**: if neither path is confident *but* the rule still
   produced a label, return that label with its (below-threshold) confidence.
   This intentionally keeps weak-signal gestures alive when ML is missing or
   unsure — downstream stages (interaction FSM) gate further on their own
   thresholds.  Returns ``(None, 0.0)`` only when the rule itself produced no
   label.

The two thresholds are independently tunable so each classifier's bar can be
raised or lowered without touching the other.

Usage::

    from gazecontrol.gesture.fusion import GestureFusion
    from gazecontrol.gesture.rule_classifier import RuleClassifier
    from gazecontrol.gesture.mlp_classifier import MLPClassifier

    fusion = GestureFusion(
        rule=RuleClassifier(),
        ml=MLPClassifier(),         # None is OK — rule-only mode
        rule_threshold=0.80,
        ml_threshold=0.70,
    )
    label, conf = fusion.classify(feature_set)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gazecontrol.gesture.classifier import GestureClassifier
    from gazecontrol.gesture.feature_extractor import FeatureSet
    from gazecontrol.gesture.rule_classifier import RuleClassifier

logger = logging.getLogger(__name__)

# Labels that the FSM knows how to handle.  Only ML outputs from this set
# are forwarded so the FSM never receives an unexpected label.
_FSM_VOCABULARY: frozenset[str] = frozenset(
    {
        "PINCH",
        "RELEASE",
        "SCROLL_UP",
        "SCROLL_DOWN",
        "SWIPE_LEFT",
        "SWIPE_RIGHT",
        "GRAB",
        "CLOSE_SIGN",
        "MAXIMIZE",
    }
)


class GestureFusion:
    """Combine a rule classifier and an ML classifier with confidence gating.

    Args:
        rule:            Rule-based classifier (required).
        ml:              Optional ML classifier.  When *None*, behaves as rule-only.
        rule_threshold:  Minimum rule confidence to accept without consulting ML.
        ml_threshold:    Minimum ML confidence to use instead of no-gesture.
        log_decisions:   If *True*, log each fusion decision at DEBUG level for
                         telemetry / distribution analysis.
    """

    def __init__(
        self,
        rule: RuleClassifier,
        ml: GestureClassifier | None = None,
        rule_threshold: float = 0.80,
        ml_threshold: float = 0.70,
        log_decisions: bool = False,
    ) -> None:
        self._rule = rule
        self._ml = ml
        self._rule_thr = rule_threshold
        self._ml_thr = ml_threshold
        self._log = log_decisions

    # ------------------------------------------------------------------

    def classify(
        self,
        features: FeatureSet | dict[str, Any] | None,
    ) -> tuple[str | None, float]:
        """Apply fusion and return ``(label, confidence)`` or ``(None, 0.0)``.

        Args:
            features: :class:`~gazecontrol.gesture.feature_extractor.FeatureSet`
                      (or legacy dict) for the current frame.

        Returns:
            ``(label, confidence)`` — ``label`` is *None* when no gesture is
            detected.
        """
        rule_label, rule_conf = self._rule.classify(features)

        # Path 1: rule is confident → accept it.
        if rule_label is not None and rule_conf >= self._rule_thr:
            if self._log:
                logger.debug("[fusion] rule=%s conf=%.2f → accepted", rule_label, rule_conf)
            return rule_label, rule_conf

        # Path 2: ML fallback when rule is unsure.
        if self._ml is not None:
            ml_label, ml_conf = self._ml.classify(features)
            if ml_label is not None and ml_conf >= self._ml_thr and ml_label in _FSM_VOCABULARY:
                if self._log:
                    logger.debug(
                        "[fusion] rule=%s conf=%.2f → ML=%s conf=%.2f → accepted",
                        rule_label,
                        rule_conf,
                        ml_label,
                        ml_conf,
                    )
                return ml_label, ml_conf

        # Path 3: neither confident — rule output if present but below threshold.
        if rule_label is not None:
            if self._log:
                logger.debug(
                    "[fusion] rule=%s conf=%.2f (below thr) → passthrough",
                    rule_label,
                    rule_conf,
                )
            return rule_label, rule_conf

        return None, 0.0
