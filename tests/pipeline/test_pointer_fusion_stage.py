"""PointerFusionStage — priority matrix tests."""

from __future__ import annotations

from gazecontrol.gaze.fixation_detector import GazeEvent
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.pipeline.pointer_fusion_stage import PointerFusionStage
from gazecontrol.settings import AppSettings


def _stage(**fusion_overrides):
    s = AppSettings()
    for k, v in fusion_overrides.items():
        setattr(s.fusion, k, v)
    return PointerFusionStage(settings=s)


def test_high_confidence_hand_wins():
    stage = _stage()
    ctx = FrameContext()
    ctx.fingertip_screen = (100, 200)
    ctx.gesture_confidence = 0.95
    ctx.gaze_screen = (500, 500)
    ctx.gaze_confidence = 0.9
    out = stage.process(ctx)
    assert out.pointer_screen == (100, 200)
    assert out.pointer_source == "hand"


def test_gaze_used_when_hand_below_threshold():
    stage = _stage(hand_confidence_threshold=0.7, gaze_confidence_threshold=0.5)
    ctx = FrameContext()
    ctx.fingertip_screen = (100, 100)
    ctx.gesture_confidence = 0.2
    ctx.gaze_screen = (400, 300)
    ctx.gaze_confidence = 0.7
    ctx.gaze_event = GazeEvent(type="saccade", point=(400, 300), velocity_px_s=900.0)
    out = stage.process(ctx)
    assert out.pointer_source == "gaze"
    assert out.pointer_screen == (400, 300)


def test_fixation_centroid_preferred_over_raw_gaze():
    stage = _stage(hand_confidence_threshold=0.99, gaze_confidence_threshold=0.5)
    ctx = FrameContext()
    ctx.gaze_screen = (400, 300)
    ctx.gaze_confidence = 0.9
    ctx.gaze_event = GazeEvent(
        type="fixation",
        point=(400, 300),
        velocity_px_s=10.0,
        centroid=(420.0, 280.0),
    )
    out = stage.process(ctx)
    assert out.pointer_source == "gaze"
    assert out.pointer_screen == (420, 280)


def test_no_pointer_when_nothing_valid():
    stage = _stage()
    ctx = FrameContext()
    out = stage.process(ctx)
    assert out.pointer_screen is None
    assert out.pointer_source == "none"


def test_low_conf_hand_used_as_last_resort():
    stage = _stage()
    ctx = FrameContext()
    ctx.fingertip_screen = (10, 20)
    ctx.gesture_confidence = 0.1
    ctx.gaze_confidence = 0.0
    out = stage.process(ctx)
    assert out.pointer_source == "hand"
    assert out.pointer_screen == (10, 20)


def test_gaze_assisted_click_when_hand_idle():
    stage = _stage(
        gaze_assisted_click=True,
        hand_confidence_threshold=0.0,
        gaze_confidence_threshold=0.5,
    )
    ctx = FrameContext()
    ctx.fingertip_screen = (50, 60)
    ctx.gesture_confidence = 0.0
    ctx.pinch_event = None
    ctx.two_finger_scroll_delta = 0
    ctx.gaze_screen = (400, 300)
    ctx.gaze_confidence = 0.9
    ctx.gaze_event = GazeEvent(
        type="fixation",
        point=(400, 300),
        velocity_px_s=10.0,
        centroid=(450.0, 350.0),
    )
    out = stage.process(ctx)
    assert out.pointer_source == "fused"
    assert out.pointer_screen == (450, 350)
