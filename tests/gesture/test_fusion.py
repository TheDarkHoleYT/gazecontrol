"""Tests for GestureFusion — confidence-weighted rule + ML fusion."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gazecontrol.gesture.feature_extractor import FeatureSet
from gazecontrol.gesture.fusion import GestureFusion


def _make_feat(**kwargs) -> FeatureSet:
    defaults = dict(
        finger_states=[1, 0, 0, 0, 0],
        finger_angles=[10.0] * 5,
        palm_direction=0.8,
        hand_velocity_x=0.0,
        hand_velocity_y=0.0,
        thumb_index_distance=0.1,
        wrist_x=0.5,
        wrist_y=0.5,
        thumb_dir_y=0.05,
    )
    defaults.update(kwargs)
    return FeatureSet(**defaults)


def _make_rule(label: str | None, conf: float) -> MagicMock:
    m = MagicMock()
    m.classify.return_value = (label, conf)
    return m


def _make_ml(label: str | None, conf: float) -> MagicMock:
    m = MagicMock()
    m.classify.return_value = (label, conf)
    return m


# ---------------------------------------------------------------------------
# Path 1: rule confident → accept rule
# ---------------------------------------------------------------------------


def test_rule_confident_accepted():
    rule = _make_rule("PINCH", 0.92)
    fusion = GestureFusion(rule=rule, ml=None, rule_threshold=0.80)
    label, conf = fusion.classify(_make_feat())
    assert label == "PINCH"
    assert conf == pytest.approx(0.92)


def test_rule_at_threshold_accepted():
    rule = _make_rule("RELEASE", 0.80)
    fusion = GestureFusion(rule=rule, ml=None, rule_threshold=0.80)
    label, _conf = fusion.classify(_make_feat())
    assert label == "RELEASE"


# ---------------------------------------------------------------------------
# Path 2: rule uncertain → ML fallback
# ---------------------------------------------------------------------------


def test_ml_fallback_when_rule_uncertain():
    rule = _make_rule("PINCH", 0.50)  # below rule_threshold
    ml = _make_ml("SCROLL_UP", 0.80)  # above ml_threshold, in FSM vocab
    fusion = GestureFusion(rule=rule, ml=ml, rule_threshold=0.80, ml_threshold=0.70)
    label, conf = fusion.classify(_make_feat())
    assert label == "SCROLL_UP"
    assert conf == pytest.approx(0.80)


def test_ml_skipped_when_label_not_in_vocabulary():
    rule = _make_rule(None, 0.0)
    ml = _make_ml("UNKNOWN_GESTURE", 0.90)  # not in FSM vocabulary
    fusion = GestureFusion(rule=rule, ml=ml, rule_threshold=0.80, ml_threshold=0.70)
    label, conf = fusion.classify(_make_feat())
    # ML filtered out; rule passthrough is None
    assert label is None
    assert conf == 0.0


def test_ml_skipped_when_below_threshold():
    rule = _make_rule(None, 0.0)
    ml = _make_ml("PINCH", 0.50)  # below ml_threshold=0.70
    fusion = GestureFusion(rule=rule, ml=ml, rule_threshold=0.80, ml_threshold=0.70)
    label, _conf = fusion.classify(_make_feat())
    assert label is None


# ---------------------------------------------------------------------------
# Path 3: rule passthrough (below threshold but label present)
# ---------------------------------------------------------------------------


def test_rule_passthrough_when_ml_also_fails():
    rule = _make_rule("GRAB", 0.60)  # below threshold
    ml = _make_ml("GRAB", 0.30)  # ML also below threshold
    fusion = GestureFusion(rule=rule, ml=ml, rule_threshold=0.80, ml_threshold=0.70)
    label, conf = fusion.classify(_make_feat())
    # Rule passthrough
    assert label == "GRAB"
    assert conf == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_none_features_returns_none():
    rule = _make_rule("PINCH", 0.92)
    rule.classify.return_value = (None, 0.0)  # rule returns None for None input
    fusion = GestureFusion(rule=rule)
    label, conf = fusion.classify(None)
    assert label is None
    assert conf == 0.0


def test_no_ml_classifier():
    rule = _make_rule("RELEASE", 0.50)  # below threshold
    fusion = GestureFusion(rule=rule, ml=None, rule_threshold=0.80)
    # No ML → falls through to passthrough
    label, _conf = fusion.classify(_make_feat())
    assert label == "RELEASE"


def test_all_fsm_vocabulary_labels_accepted():
    """Every label in _FSM_VOCABULARY must pass the ML gate."""
    from gazecontrol.gesture.fusion import _FSM_VOCABULARY

    rule = _make_rule(None, 0.0)
    for vocab_label in _FSM_VOCABULARY:
        ml = _make_ml(vocab_label, 0.85)
        fusion = GestureFusion(rule=rule, ml=ml, rule_threshold=0.80, ml_threshold=0.70)
        label, _ = fusion.classify(_make_feat())
        assert label == vocab_label, f"{vocab_label} should pass ML gate"
