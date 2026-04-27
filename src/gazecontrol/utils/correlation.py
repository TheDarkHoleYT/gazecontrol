"""Correlation context — propagates ``frame_id`` and ``run_id`` through logs.

Uses :mod:`contextvars` so per-tick state is local to the current frame
(thread + asyncio task safe).  The :class:`CorrelationFilter` injects the
context values into every :class:`logging.LogRecord`, making them available
to formatters via ``%(frame_id)s`` / ``%(run_id)s`` (text mode) or as
JSON fields (when the JSON formatter is enabled).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from contextvars import ContextVar

_frame_id: ContextVar[int] = ContextVar("gazecontrol_frame_id", default=0)


def get_frame_id() -> int:
    """Return the current frame id (0 when outside a frame context)."""
    return _frame_id.get()


def set_frame_id(value: int) -> None:
    """Set the frame id for the current context."""
    _frame_id.set(value)


@contextlib.contextmanager
def frame_context(frame_id: int) -> Iterator[None]:
    """Context manager that scopes a *frame_id* to the enclosed block."""
    token = _frame_id.set(frame_id)
    try:
        yield
    finally:
        _frame_id.reset(token)


class CorrelationFilter(logging.Filter):
    """Logging filter that injects ``frame_id`` into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add ``frame_id`` to *record* (no-op if already present)."""
        if not hasattr(record, "frame_id"):
            record.frame_id = _frame_id.get()
        return True
