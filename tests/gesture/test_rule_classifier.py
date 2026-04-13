"""Tests for RuleClassifier — geometric gesture rules."""
from __future__ import annotations

from gazecontrol.gesture.rule_classifier import RuleClassifier


def _feat(
    finger_states=None,
    finger_angles=None,
    palm_direction=0.5,
    hand_velocity_x=0.0,
    hand_velocity_y=0.0,
    thumb_index_distance=0.5,
    wrist_x=0.5,
    wrist_y=0.5,
) -> dict:
    return {
        "finger_states": finger_states or [0, 0, 0, 0, 0],
        "finger_angles": finger_angles or [0.0] * 5,
        "palm_direction": palm_direction,
        "hand_velocity_x": hand_velocity_x,
        "hand_velocity_y": hand_velocity_y,
        "thumb_index_distance": thumb_index_distance,
        "wrist_x": wrist_x,
        "wrist_y": wrist_y,
    }


def test_pinch_detected():
    rc = RuleClassifier()
    feat = _feat(
        finger_states=[0, 0, 1, 1, 1],
        thumb_index_distance=0.10,
    )
    label, conf = rc.classify(feat)
    assert label == "PINCH"
    assert conf > 0.8


def test_release_all_extended():
    rc = RuleClassifier()
    feat = _feat(finger_states=[1, 1, 1, 1, 1])
    label, _conf = rc.classify(feat)
    assert label == "RELEASE"


def test_grab_fist():
    rc = RuleClassifier()
    feat = _feat(finger_states=[0, 0, 0, 0, 0], wrist_y=0.5)
    label, _conf = rc.classify(feat)
    assert label == "GRAB"


def test_no_match_returns_none():
    rc = RuleClassifier()
    feat = _feat(finger_states=[1, 0, 1, 0, 1])
    label, conf = rc.classify(feat)
    assert label is None
    assert conf == 0.0


def test_none_input_returns_none():
    rc = RuleClassifier()
    label, conf = rc.classify(None)
    assert label is None
    assert conf == 0.0
