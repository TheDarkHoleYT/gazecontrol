"""Tests for grip_region helper."""

from __future__ import annotations

import pytest

from gazecontrol.interaction.grip_region import is_in_resize_grip


@pytest.mark.parametrize("ratio", [0.15, 0.18, 0.25])
def test_point_in_grip(ratio):
    rect = (0, 0, 100, 100)
    grip_x = int(100 * (1 - ratio))
    grip_y = int(100 * (1 - ratio))
    assert is_in_resize_grip(rect, (grip_x + 1, grip_y + 1), ratio)


@pytest.mark.parametrize("ratio", [0.15, 0.18, 0.25])
def test_point_outside_grip(ratio):
    rect = (0, 0, 100, 100)
    assert not is_in_resize_grip(rect, (5, 5), ratio)


def test_point_on_x_boundary():
    rect = (0, 0, 100, 100)
    grip_x = int(100 * 0.82)  # ratio=0.18
    assert is_in_resize_grip(rect, (grip_x, 99), 0.18)


def test_point_outside_y():
    rect = (0, 0, 100, 100)
    assert not is_in_resize_grip(rect, (95, 5), 0.18)


def test_rect_with_offset():
    rect = (200, 300, 400, 200)  # x=200, y=300, w=400, h=200
    grip_x = 200 + int(400 * 0.82)
    grip_y = 300 + int(200 * 0.82)
    assert is_in_resize_grip(rect, (grip_x + 10, grip_y + 10), 0.18)
    assert not is_in_resize_grip(rect, (200, 300), 0.18)


def test_default_ratio():
    rect = (0, 0, 100, 100)
    # Default ratio=0.18 → grip starts at x=82, y=82
    assert is_in_resize_grip(rect, (85, 85))
    assert not is_in_resize_grip(rect, (70, 70))
