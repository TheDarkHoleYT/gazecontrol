"""Tests for intent types — Enum and Action dataclass."""
from __future__ import annotations

import pytest

from gazecontrol.intent.types import (
    Action,
    ActionType,
    DragPhase,
    GestureLabel,
    IntentState,
)


class TestIntentState:
    def test_all_states_exist(self):
        states = [s.value for s in IntentState]
        assert "IDLE" in states
        assert "DRAG" in states
        assert "RESIZE" in states
        assert "COOLDOWN" in states

    def test_str_enum_comparison(self):
        assert IntentState.IDLE == "IDLE"
        assert IntentState.DRAG == "DRAG"


class TestGestureLabel:
    def test_from_str_valid(self):
        assert GestureLabel.from_str("PINCH") == GestureLabel.PINCH
        assert GestureLabel.from_str("GRAB") == GestureLabel.GRAB

    def test_from_str_none_returns_unknown(self):
        assert GestureLabel.from_str(None) == GestureLabel.UNKNOWN

    def test_from_str_invalid_returns_unknown(self):
        assert GestureLabel.from_str("FLYING_KICK") == GestureLabel.UNKNOWN

    def test_str_comparison(self):
        assert GestureLabel.PINCH == "PINCH"


class TestActionType:
    def test_all_types_str_enum(self):
        assert ActionType.DRAG == "DRAG"
        assert ActionType.CLOSE == "CLOSE"


class TestAction:
    def test_defaults_data_to_empty_dict(self):
        a = Action(type=ActionType.CLOSE, window={"hwnd": 1001})
        assert a.data == {}

    def test_frozen(self):
        a = Action(type=ActionType.CLOSE, window={"hwnd": 1001})
        with pytest.raises(Exception):  # FrozenInstanceError
            a.type = ActionType.DRAG  # type: ignore[misc]

    def test_to_legacy_dict(self):
        a = Action(type=ActionType.DRAG, window={"hwnd": 42}, data={"phase": "move"})
        d = a.to_legacy_dict()
        assert d["type"] == "DRAG"
        assert d["window"]["hwnd"] == 42
        assert d["data"]["phase"] == "move"


class TestDragPhase:
    def test_phases_exist(self):
        assert DragPhase.START == "start"
        assert DragPhase.MOVE == "move"
        assert DragPhase.END == "end"
