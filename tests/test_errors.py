"""Tests for gazecontrol.errors exception hierarchy."""

from __future__ import annotations

import pytest

from gazecontrol.errors import (
    CameraError,
    GazeControlError,
    InteractionError,
    PipelineStageError,
)

# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def test_all_subtypes_are_gazecontrol_errors():
    for cls in (CameraError, InteractionError, PipelineStageError):
        assert issubclass(cls, GazeControlError), f"{cls} must inherit GazeControlError"


def test_pipeline_stage_error_attributes():
    cause = ValueError("boom")
    err = PipelineStageError("capture", cause)
    assert err.stage_name == "capture"
    assert err.cause is cause
    assert "capture" in str(err)
    assert "boom" in str(err)


def test_catch_by_base_class():
    with pytest.raises(GazeControlError):
        raise InteractionError("bad state")


def test_catch_specific():
    with pytest.raises(InteractionError):
        raise InteractionError("bad state")


# ---------------------------------------------------------------------------
# user_message() — each subclass must return an actionable string
# ---------------------------------------------------------------------------


def test_user_message_base():
    err = GazeControlError("base message")
    assert "base message" in err.user_message()


def test_user_message_camera_error():
    msg = CameraError("cam fail").user_message()
    assert "Camera" in msg
    assert "--doctor" in msg


def test_user_message_interaction_error():
    msg = InteractionError("fsm fail").user_message()
    assert "Interaction" in msg


def test_user_message_pipeline_stage_error():
    cause = RuntimeError("boom")
    msg = PipelineStageError("gesture", cause).user_message()
    assert "gesture" in msg
    assert "--doctor" in msg


# ---------------------------------------------------------------------------
# WindowManagerError inherits from GazeControlError
# ---------------------------------------------------------------------------


def test_window_manager_error_hierarchy():
    from gazecontrol.window_manager.windows_mgr import WindowManagerError

    assert issubclass(WindowManagerError, GazeControlError)
    assert issubclass(WindowManagerError, OSError)


# ---------------------------------------------------------------------------
# exit_code_for — CLI exit-code mapping
# ---------------------------------------------------------------------------


def test_exit_code_camera_is_10():
    from gazecontrol.errors import EXIT_CAMERA, exit_code_for

    assert exit_code_for(CameraError("x")) == EXIT_CAMERA


def test_exit_code_model_load_and_download_share():
    from gazecontrol.errors import (
        EXIT_MODEL_LOAD,
        ModelDownloadError,
        ModelLoadError,
        exit_code_for,
    )

    assert exit_code_for(ModelLoadError("x")) == EXIT_MODEL_LOAD
    assert exit_code_for(ModelDownloadError("x")) == EXIT_MODEL_LOAD


def test_exit_code_unknown_gazecontrol_falls_back_to_generic():
    from gazecontrol.errors import EXIT_GENERIC, exit_code_for

    class UnknownError(GazeControlError):
        pass

    assert exit_code_for(UnknownError("x")) == EXIT_GENERIC


def test_exit_code_non_domain_is_generic():
    from gazecontrol.errors import EXIT_GENERIC, exit_code_for

    assert exit_code_for(RuntimeError("x")) == EXIT_GENERIC
