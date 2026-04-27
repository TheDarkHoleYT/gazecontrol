"""Tests for GestureLabel — canonical gesture vocabulary."""

from __future__ import annotations

from gazecontrol.gesture.labels import DEFAULT_LABELS, GestureLabel


def test_all_labels_are_strings():
    for label in GestureLabel:
        assert isinstance(label.value, str)
        assert len(label.value) > 0


def test_str_enum_compares_equal_to_string():
    """GestureLabel members must compare equal to their string value."""
    assert GestureLabel.PINCH == "PINCH"
    assert GestureLabel.SCROLL_UP == "SCROLL_UP"
    assert GestureLabel.CLOSE_SIGN == "CLOSE_SIGN"


def test_label_in_string_context():
    """StrEnum should work in f-strings and string operations."""
    assert f"{GestureLabel.PINCH}" == "PINCH"
    assert GestureLabel.MAXIMIZE.upper() == "MAXIMIZE"


def test_default_labels_contains_all_members():
    """DEFAULT_LABELS should include every GestureLabel member."""
    for label in GestureLabel:
        assert label.value in DEFAULT_LABELS


def test_default_labels_no_duplicates():
    assert len(DEFAULT_LABELS) == len(set(DEFAULT_LABELS))


def test_expected_labels_present():
    expected = {
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
    assert expected == set(DEFAULT_LABELS)


def test_lookup_by_value():
    label = GestureLabel("PINCH")
    assert label is GestureLabel.PINCH
