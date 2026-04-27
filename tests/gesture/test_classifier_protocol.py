"""Verify MLPClassifier and RuleClassifier satisfy GestureClassifier Protocol."""

from __future__ import annotations

from gazecontrol.gesture.classifier import GestureClassifier
from gazecontrol.gesture.mlp_classifier import MLPClassifier
from gazecontrol.gesture.rule_classifier import RuleClassifier


def test_rule_classifier_satisfies_protocol():
    rc = RuleClassifier()
    assert isinstance(rc, GestureClassifier), (
        "RuleClassifier does not satisfy the GestureClassifier Protocol"
    )


def test_mlp_classifier_satisfies_protocol():
    mc = MLPClassifier()
    assert isinstance(mc, GestureClassifier), (
        "MLPClassifier does not satisfy the GestureClassifier Protocol"
    )


def test_rule_classifier_classify_returns_none_on_none_input():
    rc = RuleClassifier()
    label, conf = rc.classify(None)
    assert label is None
    assert conf == 0.0


def test_mlp_classifier_classify_returns_none_when_not_loaded():
    mc = MLPClassifier()
    # Model file is absent in test environment → not loaded → classify returns (None, 0.0).
    if not mc.is_loaded():
        label, conf = mc.classify(None)
        assert label is None
        assert conf == 0.0


def test_arbitrary_object_does_not_satisfy_protocol():
    class NotAClassifier:
        pass

    assert not isinstance(NotAClassifier(), GestureClassifier)
