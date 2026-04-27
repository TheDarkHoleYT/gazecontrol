"""GestureClassifier Protocol — structural interface for gesture classifiers.

Both :class:`~gazecontrol.gesture.mlp_classifier.MLPClassifier` and
:class:`~gazecontrol.gesture.rule_classifier.RuleClassifier` satisfy this
Protocol without modification; mypy verifies the structural match at
type-check time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gazecontrol.gesture.feature_extractor import FeatureSet


@runtime_checkable
class GestureClassifier(Protocol):
    """Structural interface for hand-gesture classifiers.

    Any object that implements ``classify`` with the matching signature
    satisfies this Protocol.  No inheritance required.
    """

    def classify(self, features: FeatureSet | dict[str, Any] | None) -> tuple[str | None, float]:
        """Classify hand features into a gesture label.

        Args:
            features: A :class:`FeatureSet` instance, the legacy dict
                produced by :meth:`FeatureSet.to_dict`, or ``None`` when
                no hand is detected.

        Returns:
            ``(label, confidence)`` — ``label`` is ``None`` and confidence
            is ``0.0`` when the classifier cannot make a decision.
        """
        ...
