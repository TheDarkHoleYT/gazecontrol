"""GazeStage fallback policy (G17, ADR-0008)."""

from __future__ import annotations

import numpy as np
import pytest

from gazecontrol.errors import GazeBackendError
from gazecontrol.gaze.backend import GazePrediction
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.gaze_stage import GazeStage
from gazecontrol.settings import AppSettings, FusionSettings


def _settings(policy: str = "hand_only", threshold: int = 3) -> AppSettings:
    s = AppSettings()
    s.fusion = FusionSettings(
        hand_confidence_threshold=s.fusion.hand_confidence_threshold,
        gaze_confidence_threshold=s.fusion.gaze_confidence_threshold,
        divergence_threshold_px=s.fusion.divergence_threshold_px,
        gaze_assisted_click=s.fusion.gaze_assisted_click,
        gaze_failure_policy=policy,
        gaze_failure_threshold_frames=threshold,
    )
    return s


class _Backend:
    """Configurable stub: returns a fixed sequence of predictions / Nones / raises."""

    name = "stub"

    def __init__(self, script: list[GazePrediction | None | type[Exception]]) -> None:
        self._script = list(script)
        self._idx = 0

    def start(self) -> bool:
        return True

    def stop(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return True

    def predict(self, frame_bgr, frame_rgb, timestamp):
        if self._idx >= len(self._script):
            return None
        item = self._script[self._idx]
        self._idx += 1
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item("scripted failure")
        return item


def _ctx(t0: float = 0.0) -> FrameContext:
    ctx = FrameContext(t0=t0)
    ctx.capture_ok = True
    ctx.frame_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    ctx.frame_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    return ctx


_OK = GazePrediction(screen_xy=(500, 500), confidence=0.8, backend_name="stub")


def test_backend_down_flag_starts_false():
    stage = GazeStage(backend=_Backend([_OK]), screen_w=1920, screen_h=1080,
                      settings=_settings())
    assert stage.start() is True
    assert stage.backend_down is False
    ctx = stage.process(_ctx())
    assert ctx.gaze_backend_down is False


def test_consecutive_failures_below_threshold_do_not_flip_flag():
    stage = GazeStage(
        backend=_Backend([None, None]),
        screen_w=1920,
        screen_h=1080,
        settings=_settings(threshold=3),
    )
    stage.start()
    stage.process(_ctx())
    ctx = stage.process(_ctx())
    assert stage.backend_down is False
    assert ctx.gaze_backend_down is False


def test_threshold_failures_flip_flag_under_hand_only_policy():
    stage = GazeStage(
        backend=_Backend([None, None, None, None]),
        screen_w=1920,
        screen_h=1080,
        settings=_settings(policy="hand_only", threshold=3),
    )
    stage.start()
    for _ in range(3):
        ctx = stage.process(_ctx())
    assert stage.backend_down is True
    assert ctx.gaze_backend_down is True


def test_exceptions_count_toward_failure_streak():
    """Raised exceptions and None returns both increment the streak."""
    stage = GazeStage(
        backend=_Backend([RuntimeError, None, RuntimeError]),
        screen_w=1920,
        screen_h=1080,
        settings=_settings(policy="hand_only", threshold=3),
    )
    stage.start()
    for _ in range(3):
        ctx = stage.process(_ctx())
    assert stage.backend_down is True
    assert ctx.gaze_backend_down is True


def test_recovery_resets_streak_and_clears_flag():
    stage = GazeStage(
        backend=_Backend([None, None, None, _OK]),
        screen_w=1920,
        screen_h=1080,
        settings=_settings(policy="hand_only", threshold=3),
    )
    stage.start()
    for _ in range(3):
        stage.process(_ctx())
    assert stage.backend_down is True
    ctx = stage.process(_ctx())
    assert stage.backend_down is False
    assert ctx.gaze_backend_down is False


def test_continue_policy_does_not_flip_flag():
    """Legacy behaviour: silent backend, no degradation signal."""
    stage = GazeStage(
        backend=_Backend([None] * 10),
        screen_w=1920,
        screen_h=1080,
        settings=_settings(policy="continue", threshold=3),
    )
    stage.start()
    for _ in range(10):
        ctx = stage.process(_ctx())
    assert stage.backend_down is True  # the *internal* flag still flips ONCE...
    assert ctx.gaze_backend_down is True  # ...and ctx mirrors it for telemetry,
    # but the policy did NOT escalate (no exception, no callback).


def test_stop_policy_raises_gaze_backend_error_and_calls_callback():
    stops: list[int] = []

    def _on_stop() -> None:
        stops.append(1)

    stage = GazeStage(
        backend=_Backend([None, None, None]),
        screen_w=1920,
        screen_h=1080,
        settings=_settings(policy="stop", threshold=3),
        on_stop_requested=_on_stop,
    )
    stage.start()
    stage.process(_ctx())
    stage.process(_ctx())
    with pytest.raises(GazeBackendError, match="silent for 3 frames"):
        stage.process(_ctx())
    assert stops == [1]


def test_stop_policy_failure_in_callback_does_not_swallow_error():
    def _on_stop() -> None:
        raise RuntimeError("callback exploded")

    stage = GazeStage(
        backend=_Backend([None, None]),
        screen_w=1920,
        screen_h=1080,
        settings=_settings(policy="stop", threshold=2),
        on_stop_requested=_on_stop,
    )
    stage.start()
    stage.process(_ctx())
    # The callback explodes but we still propagate the GazeBackendError —
    # operators must learn about the policy outcome.
    with pytest.raises(GazeBackendError):
        stage.process(_ctx())


def test_threshold_one_fires_immediately():
    stage = GazeStage(
        backend=_Backend([None]),
        screen_w=1920,
        screen_h=1080,
        settings=_settings(policy="hand_only", threshold=1),
    )
    stage.start()
    ctx = stage.process(_ctx())
    assert ctx.gaze_backend_down is True


def test_alternating_failure_and_success_keeps_flag_clear():
    """A single missed frame between successes should not degrade —
    only a *streak* of failures does."""
    stage = GazeStage(
        backend=_Backend([None, _OK, None, _OK, None, _OK]),
        screen_w=1920,
        screen_h=1080,
        settings=_settings(policy="hand_only", threshold=2),
    )
    stage.start()
    for _ in range(6):
        ctx = stage.process(_ctx())
    assert stage.backend_down is False
    assert ctx.gaze_backend_down is False
