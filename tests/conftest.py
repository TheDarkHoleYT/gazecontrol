"""Pytest configuration and shared fixtures for gazecontrol tests."""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Generator

import numpy as np
import pytest
from hypothesis import HealthCheck
from hypothesis import settings as hyp_settings

from gazecontrol.settings import AppSettings, reset_settings
from tests.helpers import FakeVideoCapture, make_fake_hand_result


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip @pytest.mark.win32 tests on non-Windows platforms."""
    if sys.platform != "win32":
        skip_win32 = pytest.mark.skip(reason="Win32-only test (ctypes.windll absent)")
        for item in items:
            if "win32" in item.keywords:
                item.add_marker(skip_win32)

# ---------------------------------------------------------------------------
# Determinism — pin all random sources before any test imports modules that
# spin up classifiers or generate sample data.
# ---------------------------------------------------------------------------

os.environ.setdefault("PYTHONHASHSEED", "0")

hyp_settings.register_profile(
    "ci",
    deadline=None,
    max_examples=50,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
hyp_settings.load_profile("ci")


@pytest.fixture(autouse=True)
def _pin_random_seeds() -> Generator[None, None, None]:
    """Reset ``random`` and ``numpy.random`` to a fixed seed for each test."""
    random.seed(0)
    np.random.seed(0)
    yield


# ---------------------------------------------------------------------------
# Settings override fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def test_settings() -> Generator[AppSettings, None, None]:
    """Provide a clean AppSettings instance for each test and reset after."""
    s = AppSettings()
    reset_settings(s)
    yield s
    reset_settings(None)


# ---------------------------------------------------------------------------
# Fixture wrappers for helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_video_capture():
    """Return the FakeVideoCapture class for use in tests."""
    return FakeVideoCapture


@pytest.fixture
def fake_hand_result():
    """Return the make_fake_hand_result factory for use in tests."""
    return make_fake_hand_result
