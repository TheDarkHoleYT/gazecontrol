"""Per-app fusion overrides — PointerFusionStage._cfg() (G24)."""

from __future__ import annotations

import numpy as np
import pytest

from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.pointer_fusion_stage import PointerFusionStage
from gazecontrol.settings import AppSettings, FusionSettings


def _settings(**overrides) -> AppSettings:
    s = AppSettings()
    if overrides:
        s.fusion = FusionSettings(
            hand_confidence_threshold=overrides.get(
                "hand_confidence_threshold", s.fusion.hand_confidence_threshold
            ),
            gaze_confidence_threshold=overrides.get(
                "gaze_confidence_threshold", s.fusion.gaze_confidence_threshold
            ),
            divergence_threshold_px=overrides.get(
                "divergence_threshold_px", s.fusion.divergence_threshold_px
            ),
            gaze_assisted_click=s.fusion.gaze_assisted_click,
            gaze_failure_policy=s.fusion.gaze_failure_policy,
            gaze_failure_threshold_frames=s.fusion.gaze_failure_threshold_frames,
            app_overrides=overrides.get("app_overrides", s.fusion.app_overrides),
        )
    return s


def _ctx(foreground_app: str | None = None, **kwargs) -> FrameContext:
    ctx = FrameContext()
    ctx.capture_ok = True
    ctx.frame_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    ctx.frame_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    ctx.foreground_app = foreground_app
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    return ctx


def test_cfg_returns_base_when_no_foreground_app():
    stage = PointerFusionStage(settings=_settings())
    cfg = stage._cfg(_ctx(foreground_app=None))
    assert cfg.gaze_confidence_threshold == 0.6  # default


def test_cfg_returns_base_when_app_has_no_override():
    s = _settings(app_overrides={"firefox.exe": {"gaze_confidence_threshold": 0.9}})
    stage = PointerFusionStage(settings=s)
    cfg = stage._cfg(_ctx(foreground_app="figma.exe"))
    # figma.exe is not in the override map → base value preserved.
    assert cfg.gaze_confidence_threshold == 0.6


def test_cfg_applies_matching_override():
    s = _settings(app_overrides={"figma.exe": {"gaze_confidence_threshold": 0.9}})
    stage = PointerFusionStage(settings=s)
    cfg = stage._cfg(_ctx(foreground_app="figma.exe"))
    assert cfg.gaze_confidence_threshold == 0.9
    # Other fields untouched.
    assert cfg.hand_confidence_threshold == 0.7


def test_cfg_partial_override_leaves_other_fields_alone():
    """Per the contract: only the keys the operator listed change."""
    s = _settings(
        app_overrides={
            "figma.exe": {
                "gaze_confidence_threshold": 0.8,
                "divergence_threshold_px": 100.0,
            }
        }
    )
    stage = PointerFusionStage(settings=s)
    cfg = stage._cfg(_ctx(foreground_app="figma.exe"))
    assert cfg.gaze_confidence_threshold == 0.8
    assert cfg.divergence_threshold_px == 100.0
    assert cfg.hand_confidence_threshold == 0.7  # untouched


def test_cfg_ignores_unknown_override_keys():
    """A typo / removed field in the override must not crash the pipeline."""
    s = _settings(app_overrides={"figma.exe": {"made_up_key": 1.23}})
    stage = PointerFusionStage(settings=s)
    cfg = stage._cfg(_ctx(foreground_app="figma.exe"))
    assert not hasattr(cfg, "made_up_key")
    # Base config untouched.
    assert cfg.gaze_confidence_threshold == 0.6


def test_app_overrides_schema_rejects_nested_dict():
    """app_overrides is typed dict[str, dict[str, float]] — nested
    dict values are rejected at settings construction, not deferred to
    runtime. This is defence in depth: the loader catches the typo
    before the pipeline starts."""
    from pydantic_core import ValidationError

    with pytest.raises(ValidationError):
        FusionSettings(app_overrides={"figma.exe": {"k": {"nested": 1.0}}})  # type: ignore[arg-type]


def test_override_affects_fusion_decision_end_to_end():
    """A high gaze_confidence_threshold in an app override must cause a
    gaze sample below that bar to lose to a (weak) hand sample."""
    s = _settings(app_overrides={"figma.exe": {"gaze_confidence_threshold": 0.95}})
    stage = PointerFusionStage(settings=s)
    stage.start()

    ctx = _ctx(foreground_app="figma.exe")
    # Hand: present but weak (below the 0.7 threshold).
    ctx.fingertip_screen = (200, 200)
    ctx.gesture_confidence = 0.3
    # Gaze: present with moderate confidence (0.7) — would have won
    # under the default 0.6 threshold; the figma override bumps it to
    # 0.95 so the gaze drops out and the low-conf hand inherits via
    # the fallback branch.
    ctx.gaze_screen = (500, 500)
    ctx.gaze_confidence = 0.7

    out = stage.process(ctx)

    assert out.pointer_screen == (200, 200)
    assert out.pointer_source == "hand"


def test_no_override_falls_back_to_base_decision():
    """Sanity: without an override, the gaze sample wins per the
    legacy threshold of 0.6."""
    s = _settings()
    stage = PointerFusionStage(settings=s)
    stage.start()

    ctx = _ctx(foreground_app=None)
    ctx.fingertip_screen = (200, 200)
    ctx.gesture_confidence = 0.3
    ctx.gaze_screen = (500, 500)
    ctx.gaze_confidence = 0.7

    out = stage.process(ctx)

    # Gaze beats the low-confidence hand path under default thresholds.
    assert out.pointer_screen == (500, 500)
    assert out.pointer_source == "gaze"


def test_foreground_app_helper_returns_none_off_windows(monkeypatch):
    import sys

    from gazecontrol.runtime.foreground_app import detect_foreground_app

    monkeypatch.setattr(sys, "platform", "linux")
    assert detect_foreground_app() is None
