"""Gaze tracking subsystem.

Public surface:
    - :class:`GazeBackend` Protocol — abstraction over eye tracker backends.
    - :class:`GazePrediction` — value object returned by every backend.
    - :class:`EyetraxBackend`, :class:`L2CSBackend`, :class:`EnsembleBackend`.
    - :class:`GazeMapper` — angle → screen coordinate regressor.
    - :class:`FaceCropper` — preprocessing for L2CS-Net.
    - :class:`FixationDetector`, :class:`GazeEvent` — I-VT classifier.
    - :class:`DriftCorrector` — implicit recalibration.
"""

from gazecontrol.gaze.backend import GazeBackend, GazePrediction
from gazecontrol.gaze.drift_corrector import DriftCorrector
from gazecontrol.gaze.face_crop import FaceCropper
from gazecontrol.gaze.fixation_detector import FixationDetector, GazeEvent
from gazecontrol.gaze.gaze_mapper import GazeMapper

__all__ = [
    "DriftCorrector",
    "FaceCropper",
    "FixationDetector",
    "GazeBackend",
    "GazeEvent",
    "GazeMapper",
    "GazePrediction",
]
