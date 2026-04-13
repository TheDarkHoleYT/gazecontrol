"""Pytest configuration and shared fixtures for gazecontrol tests."""
import sys
import types
import pytest


@pytest.fixture(autouse=False)
def mock_config():
    """Install a minimal mock gazecontrol.config into sys.modules.

    Use this fixture in tests that import modules relying on gazecontrol.config
    *before* the real config is importable (e.g., tests for state_machine that
    need specific threshold values).

    The fixture tears down by restoring the previous sys.modules state.
    """
    real = sys.modules.get("gazecontrol.config")
    cfg = types.ModuleType("gazecontrol.config")
    cfg.DWELL_TIME_MS = 400
    cfg.READY_TIMEOUT_S = 3.0
    cfg.COOLDOWN_MS = 300
    cfg.GESTURE_CONFIDENCE_THRESHOLD = 0.85
    cfg.DRAG_HAND_SENSITIVITY = 1.5
    cfg.RESIZE_HAND_SENSITIVITY = 2.0
    # Also register under the bare name so any residual bare import still works
    real_bare = sys.modules.get("config")
    sys.modules["gazecontrol.config"] = cfg
    sys.modules["config"] = cfg
    yield cfg
    if real is None:
        sys.modules.pop("gazecontrol.config", None)
    else:
        sys.modules["gazecontrol.config"] = real
    if real_bare is None:
        sys.modules.pop("config", None)
    else:
        sys.modules["config"] = real_bare
