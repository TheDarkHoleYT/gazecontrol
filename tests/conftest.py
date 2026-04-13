"""Pytest configuration and shared fixtures for gazecontrol tests."""
from __future__ import annotations

from collections.abc import Generator

import pytest

from gazecontrol.settings import AppSettings, reset_settings
from tests.helpers import FakeVideoCapture, make_fake_hand_result

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
