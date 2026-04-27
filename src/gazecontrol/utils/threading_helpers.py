"""Small threading utilities used by the pipeline + capture stages.

- :class:`ShutdownToken` — cooperative shutdown signal that propagates
  through nested workers (wraps :class:`threading.Event` with a friendly API).
- :func:`run_with_timeout` — run a callable on a daemon thread and raise
  :class:`TimeoutError` if it does not finish within the budget.  Useful
  to bound calls into untyped C-extensions (cv2.read, mediapipe init).

Neither helper requires any third-party dependency.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ShutdownToken:
    """Cooperative shutdown signal — wraps an Event with explicit semantics."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self) -> None:
        """Mark the token as 'shutdown requested'.  Idempotent."""
        self._event.set()

    def is_set(self) -> bool:
        """True once :meth:`request` has been called."""
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until shutdown requested.  Returns True when set."""
        return self._event.wait(timeout)


class _Result(Generic[T]):
    __slots__ = ("error", "value")

    def __init__(self) -> None:
        self.value: T | None = None
        self.error: BaseException | None = None


def run_with_timeout(
    fn: Callable[..., T],
    *args: Any,
    timeout: float,
    **kwargs: Any,
) -> T:
    """Run *fn* on a daemon thread and raise :class:`TimeoutError` past *timeout*.

    The worker thread is left running on timeout — daemon=True ensures it
    cannot block process exit, and the caller treats the call as failed.
    Use only for I/O-bound work where leaking a thread on stall is acceptable
    (camera reads, model loads).

    Raises:
        TimeoutError: when the worker has not produced a result within
            ``timeout`` seconds.
        Any exception that *fn* raises (re-raised on the caller's thread).
    """
    result: _Result[T] = _Result()

    def _runner() -> None:
        try:
            result.value = fn(*args, **kwargs)
        except BaseException as exc:
            result.error = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"{fn.__qualname__} exceeded {timeout:.1f}s budget")
    if result.error is not None:
        raise result.error
    # _Result.value is set on success; mypy can't see the runtime guarantee.
    return result.value  # type: ignore[return-value]
