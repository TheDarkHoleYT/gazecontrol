"""Tests for utils.correlation — frame_id propagation through logging."""

from __future__ import annotations

import logging

from gazecontrol.utils.correlation import (
    CorrelationFilter,
    frame_context,
    get_frame_id,
    set_frame_id,
)


def test_default_frame_id_is_zero():
    set_frame_id(0)
    assert get_frame_id() == 0


def test_set_and_get_frame_id():
    set_frame_id(42)
    assert get_frame_id() == 42
    set_frame_id(0)


def test_frame_context_scopes_value():
    set_frame_id(0)
    with frame_context(7):
        assert get_frame_id() == 7
    assert get_frame_id() == 0


def test_correlation_filter_injects_frame_id():
    set_frame_id(0)
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="x",
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    with frame_context(99):
        CorrelationFilter().filter(record)
    assert record.frame_id == 99


def test_correlation_filter_preserves_existing_frame_id():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="x",
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.frame_id = 123
    CorrelationFilter().filter(record)
    assert record.frame_id == 123
