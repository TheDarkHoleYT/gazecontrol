"""Snapshot tests guarding the public API surface (ADR-0006).

If any of these tests fail, either you intentionally added/removed a
public symbol — update the snapshot AND the matching ``__all__`` —
or you accidentally exposed an internal.  The exposure is the bug.
"""

from __future__ import annotations

import importlib

import pytest

EXPECTED: dict[str, set[str]] = {
    "gazecontrol": {"__version__", "AppSettings", "InputMode", "get_settings"},
    "gazecontrol.pipeline": {
        "FrameContext",
        "GazeControlPipeline",
        "PipelineEngine",
        "PipelineStage",
        "QtPipelineThread",
    },
    "gazecontrol.gaze": {
        "DriftCorrector",
        "FaceCropper",
        "FixationDetector",
        "GazeBackend",
        "GazeEvent",
        "GazeMapper",
        "GazePrediction",
    },
    "gazecontrol.gesture": {
        "FeatureSet",
        "GestureClassifier",
        "GestureFeatureExtractor",
        "HandDetector",
        "MLPClassifier",
        "RuleClassifier",
        "TCNClassifier",
    },
    "gazecontrol.runtime": {"InputMode", "load_persisted_mode", "persist_mode"},
}


@pytest.mark.parametrize("module_name,expected", sorted(EXPECTED.items()))
def test_public_all_matches_snapshot(module_name: str, expected: set[str]):
    mod = importlib.import_module(module_name)
    assert set(getattr(mod, "__all__", [])) == expected, (
        f"{module_name}.__all__ drifted from the public-API snapshot.\n"
        f"  Expected: {sorted(expected)}\n"
        f"  Got:      {sorted(getattr(mod, '__all__', []))}"
    )


def test_top_level_version_is_string():
    import gazecontrol

    assert isinstance(gazecontrol.__version__, str)
    assert gazecontrol.__version__.count(".") == 2
