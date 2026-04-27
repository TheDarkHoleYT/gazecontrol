"""Smoke tests for PipelineFactory mode → stage list."""

from __future__ import annotations

import pytest

from gazecontrol.runtime.input_mode import InputMode

# WindowsManager / FingertipMapper require Win32 + MediaPipe; skip when not Win32.
pytest.importorskip("win32gui", reason="PipelineFactory requires Windows API")


def _build(mode):
    from gazecontrol.runtime.pipeline_factory import PipelineFactory

    factory = PipelineFactory(mode=mode, vdesk=(0, 0, 1920, 1080))
    return factory.build()


@pytest.mark.win32
def test_hand_only_stages():
    built = _build(InputMode.HAND_ONLY)
    names = [getattr(s, "name", "?") for s in built.engine._stages]
    assert names == ["capture", "gesture", "interaction", "action"]
    assert built.gaze_stage is None


@pytest.mark.win32
def test_eye_hand_stages():
    built = _build(InputMode.EYE_HAND)
    names = [getattr(s, "name", "?") for s in built.engine._stages]
    assert names == ["capture", "gaze", "gesture", "pointer_fusion", "interaction", "action"]
    assert built.gaze_stage is not None
