"""EnsembleBackend tests — weight blending, fallback, blink propagation."""

from __future__ import annotations

import numpy as np
import pytest

from gazecontrol.gaze.backend import GazePrediction, GazeQuality
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


def test_static_mode_blends_at_fixed_weights():
    """Legacy v0.7–v0.8 behaviour: weights are the configured base values,
    ignoring per-frame confidence. Pinned via ``mode="static"``."""
    a = _FakeBackend("a", _make_pred(xy=(100, 100), confidence=1.0))
    b = _FakeBackend("b", _make_pred(xy=(200, 200), confidence=0.5))
    ens = EnsembleBackend(
        primary=a,
        secondary=b,
        weight_primary=0.3,
        weight_secondary=0.7,
        mode="static",
    )
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


# ---------------------------------------------------------------------------
# v1.0 G3 — confidence-weighted and Kalman ensemble modes
# ---------------------------------------------------------------------------


def _pred(xy, conf, *, uncertainty_px=None, head_pose=None, face_id=None, quality=0):
    return GazePrediction(
        screen_xy=xy,
        confidence=conf,
        backend_name="fake",
        uncertainty_px=uncertainty_px,
        head_pose_rad=head_pose,
        face_id=face_id,
        quality_flags=quality,
    )


def test_default_mode_is_confidence_weighted():
    """The v1.0 default puts more weight on whichever backend is more sure
    of itself this frame, regardless of the base weights."""
    a = _FakeBackend("a", _pred((100, 100), 1.0))
    b = _FakeBackend("b", _pred((200, 200), 0.5))
    ens = EnsembleBackend(primary=a, secondary=b, weight_primary=0.3, weight_secondary=0.7)
    ens.start()
    pred = ens.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0)
    assert pred is not None
    # w1 = 0.3*1.0 / (0.3*1.0 + 0.7*0.5) = 0.3/0.65 ≈ 0.4615
    # x = 0.4615*100 + 0.5385*200 ≈ 153.85 → round = 154
    assert pred.screen_xy == (154, 154)
    assert pred.confidence == pytest.approx(0.4615 * 1.0 + 0.5385 * 0.5, abs=1e-2)
    # Static mode on the same inputs would give 170 — confirm the modes differ.
    static = EnsembleBackend(primary=a, secondary=b, weight_primary=0.3, weight_secondary=0.7,
                             mode="static")
    static.start()
    s_pred = static.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0)
    assert s_pred is not None and s_pred.screen_xy == (170, 170)
    assert pred.screen_xy != s_pred.screen_xy


def test_confidence_mode_drops_backend_with_zero_confidence():
    """A backend that reports zero confidence must not move the blended
    point — only the other one's prediction survives."""
    a = _FakeBackend("a", _pred((100, 100), 0.0))
    b = _FakeBackend("b", _pred((200, 200), 0.8))
    ens = EnsembleBackend(primary=a, secondary=b, weight_primary=0.5, weight_secondary=0.5,
                         mode="confidence")
    ens.start()
    pred = ens.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0)
    assert pred is not None and pred.screen_xy == (200, 200)


def test_confidence_mode_falls_back_to_base_weights_when_both_zero():
    """When both backends report zero confidence we keep emitting a sample
    using the static base weights — losing the pointer is worse than a
    mathematically pure fallback."""
    a = _FakeBackend("a", _pred((0, 0), 0.0))
    b = _FakeBackend("b", _pred((100, 100), 0.0))
    ens = EnsembleBackend(primary=a, secondary=b, weight_primary=0.3, weight_secondary=0.7,
                         mode="confidence")
    ens.start()
    pred = ens.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0)
    assert pred is not None
    # 0.3*0 + 0.7*100 = 70
    assert pred.screen_xy == (70, 70)


def test_kalman_mode_variance_weighted_blend():
    """Variance-weighted LS: smaller sigma → larger weight, with the fused
    sigma equal to sqrt(1 / (1/σ1² + 1/σ2²))."""
    a = _FakeBackend("a", _pred((0, 0), 1.0, uncertainty_px=10.0))
    b = _FakeBackend("b", _pred((100, 100), 1.0, uncertainty_px=20.0))
    ens = EnsembleBackend(primary=a, secondary=b, weight_primary=0.5, weight_secondary=0.5,
                         mode="kalman")
    ens.start()
    pred = ens.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0)
    assert pred is not None
    # 1/100 vs 1/400 → 0.8 / 0.2
    # x = 0.8*0 + 0.2*100 = 20
    assert pred.screen_xy == (20, 20)
    # fused sigma = sqrt(1 / (1/100 + 1/400)) = sqrt(80) ≈ 8.944
    assert pred.uncertainty_px == pytest.approx(8.944, rel=1e-3)


def test_kalman_falls_back_to_confidence_when_uncertainty_missing():
    """When at least one prediction lacks uncertainty, Kalman is undefined
    — degrade to ``"confidence"`` for that frame rather than erroring."""
    a = _FakeBackend("a", _pred((100, 100), 1.0))  # no uncertainty
    b = _FakeBackend("b", _pred((200, 200), 0.5, uncertainty_px=15.0))
    ens = EnsembleBackend(primary=a, secondary=b, weight_primary=0.3, weight_secondary=0.7,
                         mode="kalman")
    ens.start()
    pred = ens.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0)
    # Should match the confidence-mode answer from earlier in this file.
    assert pred is not None and pred.screen_xy == (154, 154)


def test_invalid_mode_rejected():
    a = _FakeBackend("a", None)
    b = _FakeBackend("b", None)
    with pytest.raises(ValueError):
        EnsembleBackend(primary=a, secondary=b, mode="bogus")  # type: ignore[arg-type]


def test_v10_enrichment_fields_propagate_through_blend():
    """head_pose / face_id come from the dominant (more-weighted)
    backend; quality_flags are OR'd across both inputs so no signal is
    lost downstream."""
    a = _FakeBackend(
        "a",
        _pred((100, 100), 0.2, head_pose=(0.1, 0.0, 0.0), face_id=1,
              quality=int(GazeQuality.OFF_AXIS)),
    )
    b = _FakeBackend(
        "b",
        _pred((200, 200), 0.9, head_pose=(0.3, 0.1, 0.0), face_id=2,
              quality=int(GazeQuality.MULTI_FACE)),
    )
    ens = EnsembleBackend(primary=a, secondary=b, weight_primary=0.5, weight_secondary=0.5,
                         mode="confidence")
    ens.start()
    pred = ens.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0)
    assert pred is not None
    # b dominates (conf 0.9 vs 0.2 with equal base weights).
    assert pred.head_pose_rad == (0.3, 0.1, 0.0)
    assert pred.face_id == 2
    # Quality flags from both backends are merged.
    assert pred.quality_flags & GazeQuality.OFF_AXIS
    assert pred.quality_flags & GazeQuality.MULTI_FACE


def test_single_backend_passthrough_preserves_v10_fields():
    """When only one backend produces a prediction, its v1.0 fields
    must survive the ensemble's re-wrap."""
    a = _FakeBackend("a", None)
    b = _FakeBackend(
        "b",
        _pred((42, 42), 0.7, uncertainty_px=12.5, head_pose=(0.1, 0.2, 0.3),
              face_id=7, quality=int(GazeQuality.OCCLUDED)),
    )
    ens = EnsembleBackend(primary=a, secondary=b)
    ens.start()
    pred = ens.predict(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), 0.0)
    assert pred is not None
    assert pred.screen_xy == (42, 42)
    assert pred.uncertainty_px == 12.5
    assert pred.head_pose_rad == (0.1, 0.2, 0.3)
    assert pred.face_id == 7
    assert pred.quality_flags & GazeQuality.OCCLUDED
    assert pred.backend_name == "ensemble"
