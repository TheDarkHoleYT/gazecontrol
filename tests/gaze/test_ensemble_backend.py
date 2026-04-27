"""EnsembleBackend tests — weight blending, fallback, blink propagation."""

from __future__ import annotations

import numpy as np
import pytest

from gazecontrol.gaze.backend import GazePrediction
from gazecontrol.gaze.ensemble_backend import EnsembleBackend


class _FakeBackend:
    def __init__(
        self,
        name: str,
        prediction: GazePrediction | None,
        *,
        start_ok: bool = True,
        calibrated: bool = True,
    ) -> None:
        self.name = name
        self._pred = prediction
        self._start_ok = start_ok
        self._calibrated = calibrated
        self.started = False
        self.stopped = False

    def start(self) -> bool:
        self.started = True
        return self._start_ok

    def stop(self) -> None:
        self.stopped = True

    def is_calibrated(self) -> bool:
        return self._calibrated

    def predict(self, frame_bgr, frame_rgb, timestamp):
        return self._pred


def _make_pred(xy=(100, 200), confidence=0.8, blink=False):
    return GazePrediction(
        screen_xy=xy,
        confidence=confidence,
        blink=blink,
        backend_name="fake",
    )


def test_blend_two_valid_predictions_weighted_average():
    a = _FakeBackend("a", _make_pred(xy=(100, 100), confidence=1.0))
    b = _FakeBackend("b", _make_pred(xy=(200, 200), confidence=0.5))
    ens = EnsembleBackend(primary=a, secondary=b, weight_primary=0.3, weight_secondary=0.7)
    assert ens.start() is True
    pred = ens.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0)
    assert pred is not None
    # 0.3*100 + 0.7*200 = 170 ; same for y.
    assert pred.screen_xy == (170, 170)
    # Weighted confidence: 0.3*1.0 + 0.7*0.5 = 0.65
    assert pred.confidence == pytest.approx(0.65, rel=1e-6)
    assert pred.backend_name == "ensemble"


def test_blend_returns_secondary_when_primary_missing():
    a = _FakeBackend("a", None)
    b = _FakeBackend("b", _make_pred(xy=(50, 60)))
    ens = EnsembleBackend(primary=a, secondary=b, weight_primary=0.3, weight_secondary=0.7)
    ens.start()
    pred = ens.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0)
    assert pred is not None and pred.screen_xy == (50, 60)


def test_blend_returns_none_when_both_missing():
    a = _FakeBackend("a", None)
    b = _FakeBackend("b", None)
    ens = EnsembleBackend(primary=a, secondary=b)
    ens.start()
    assert ens.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0) is None


def test_blink_propagates():
    a = _FakeBackend("a", _make_pred(blink=True))
    b = _FakeBackend("b", _make_pred(xy=(100, 100)))
    ens = EnsembleBackend(primary=a, secondary=b)
    ens.start()
    pred = ens.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0)
    assert pred is not None and pred.blink is True


def test_start_fails_only_when_both_fail():
    a = _FakeBackend("a", None, start_ok=False)
    b = _FakeBackend("b", None, start_ok=False)
    ens = EnsembleBackend(primary=a, secondary=b)
    assert ens.start() is False


def test_start_succeeds_when_one_starts():
    a = _FakeBackend("a", None, start_ok=False)
    b = _FakeBackend("b", None, start_ok=True)
    ens = EnsembleBackend(primary=a, secondary=b)
    assert ens.start() is True


def test_negative_weight_rejected():
    a = _FakeBackend("a", None)
    b = _FakeBackend("b", None)
    with pytest.raises(ValueError):
        EnsembleBackend(primary=a, secondary=b, weight_primary=-0.1, weight_secondary=1.0)


def test_zero_weights_rejected():
    a = _FakeBackend("a", None)
    b = _FakeBackend("b", None)
    with pytest.raises(ValueError):
        EnsembleBackend(primary=a, secondary=b, weight_primary=0.0, weight_secondary=0.0)


def test_is_calibrated_or_semantics():
    a = _FakeBackend("a", None, calibrated=False)
    b = _FakeBackend("b", None, calibrated=True)
    ens = EnsembleBackend(primary=a, secondary=b)
    assert ens.is_calibrated() is True


def test_stop_propagates_to_both():
    a = _FakeBackend("a", None)
    b = _FakeBackend("b", None)
    ens = EnsembleBackend(primary=a, secondary=b)
    ens.start()
    ens.stop()
    assert a.stopped and b.stopped


def test_predict_failure_in_one_backend_returns_other():
    class _RaisingBackend(_FakeBackend):
        def predict(self, frame_bgr, frame_rgb, timestamp):
            raise RuntimeError("boom")

    a = _RaisingBackend("a", _make_pred())
    b = _FakeBackend("b", _make_pred(xy=(99, 88)))
    ens = EnsembleBackend(primary=a, secondary=b)
    ens.start()
    pred = ens.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0)
    assert pred is not None and pred.screen_xy == (99, 88)
