"""GazeControl exception hierarchy.

All domain-specific exceptions inherit from ``GazeControlError`` so callers
can catch the whole family with a single ``except GazeControlError`` clause,
or individual subtypes for finer-grained handling.

Usage::

    from gazecontrol.errors import InteractionError

    try:
        fsm.update(...)
    except InteractionError as exc:
        logger.error("Interaction failed: %s", exc)
"""

from __future__ import annotations


class GazeControlError(Exception):
    """Base class for all GazeControl runtime errors."""

    def user_message(self) -> str:
        """Return an actionable, human-readable error description.

        Subclasses override this to provide specific troubleshooting guidance.
        The default falls back to the exception's string representation.
        """
        return str(self)


class CameraError(GazeControlError):
    """Raised when the webcam cannot be opened or fails unrecoverably."""

    def user_message(self) -> str:
        return (
            f"Camera error: {self}\n"
            "  • Make sure the webcam is connected and no other app is using it.\n"
            "  • Close: Windows Camera, Teams, Zoom, Skype, Discord, OBS.\n"
            "  • Run 'gazecontrol --doctor' to probe camera status."
        )


class InteractionError(GazeControlError):
    """Raised when the interaction subsystem encounters an unrecoverable error."""

    def user_message(self) -> str:
        return (
            f"Interaction error: {self}\n"
            "  • Check logs for details.\n"
            "  • Make sure your hand is visible to the webcam."
        )


class GazeBackendError(GazeControlError):
    """Raised when a gaze backend fails to load, predict, or shut down."""

    def user_message(self) -> str:
        return (
            f"Gaze backend error: {self}\n"
            "  • Run 'gazecontrol --doctor' to verify the eye-tracking stack.\n"
            "  • Make sure the L2CS model and a calibration profile are present.\n"
            "  • Re-run calibration: 'gazecontrol --calibrate-gaze'."
        )


class CalibrationError(GazeControlError):
    """Raised when gaze calibration cannot complete."""

    def user_message(self) -> str:
        return (
            f"Calibration error: {self}\n"
            "  • Make sure you are in front of the webcam, well lit.\n"
            "  • Avoid backlighting and reflections on glasses.\n"
            "  • Retry with: 'gazecontrol --calibrate-gaze --profile default'."
        )


class ModelLoadError(GazeControlError):
    """Raised when an ML model file cannot be loaded from disk."""

    def user_message(self) -> str:
        return (
            f"Model load error: {self}\n"
            "  • Run 'gazecontrol --doctor' to verify model files.\n"
            "  • Re-download with the bundled model_downloader."
        )


class ModelDownloadError(ModelLoadError):
    """Raised when an ML model download fails after all retry attempts."""

    def user_message(self) -> str:
        return (
            f"Model download error: {self}\n"
            "  • Check your network connection.\n"
            "  • Retry; downloads back-off exponentially across 3 attempts.\n"
            "  • Set GAZECONTROL_ALLOW_UNPINNED_MODELS=1 only for debugging."
        )


class PipelineStageError(GazeControlError):
    """Wraps an exception raised by a pipeline stage during ``process()``.

    Attributes:
        stage_name: Name of the stage that failed.
        cause:      The original exception.
    """

    def __init__(self, stage_name: str, cause: Exception) -> None:
        self.stage_name = stage_name
        self.cause = cause
        super().__init__(f"stage {stage_name!r} failed: {cause}")

    def user_message(self) -> str:
        return (
            f"Pipeline stage '{self.stage_name}' failed: {self.cause}\n"
            "  • Check logs for the full traceback.\n"
            "  • Run 'gazecontrol --doctor' to verify hardware."
        )


# ---------------------------------------------------------------------------
# Exit code mapping for ``gazecontrol`` CLI
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_CAMERA = 10
EXIT_MODEL_LOAD = 11
EXIT_GAZE_BACKEND = 12
EXIT_CALIBRATION = 13
EXIT_INTERACTION = 14
EXIT_PIPELINE_STAGE = 15

_EXIT_BY_TYPE: dict[type[GazeControlError], int] = {
    CameraError: EXIT_CAMERA,
    ModelLoadError: EXIT_MODEL_LOAD,
    ModelDownloadError: EXIT_MODEL_LOAD,
    GazeBackendError: EXIT_GAZE_BACKEND,
    CalibrationError: EXIT_CALIBRATION,
    InteractionError: EXIT_INTERACTION,
    PipelineStageError: EXIT_PIPELINE_STAGE,
}


def exit_code_for(exc: BaseException) -> int:
    """Map a GazeControl exception to its CLI exit code.

    Returns ``EXIT_GENERIC`` for unrecognised :class:`GazeControlError` subtypes
    and any non-domain exception.
    """
    if isinstance(exc, GazeControlError):
        for cls, code in _EXIT_BY_TYPE.items():
            if isinstance(exc, cls):
                return code
        return EXIT_GENERIC
    return EXIT_GENERIC
