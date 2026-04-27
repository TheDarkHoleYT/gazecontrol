"""Gesture detection + classification subsystem.

Public surface:

- :class:`GestureFeatureExtractor` — landmark → :class:`FeatureSet` features.
- :class:`FeatureSet` — typed container for one frame's features.
- :class:`HandDetector` — MediaPipe Tasks-API hand landmark detector.
- :class:`GestureClassifier` (Protocol) — structural interface for classifiers.
- :class:`RuleClassifier` — geometric heuristic classifier.
- :class:`MLPClassifier`, :class:`TCNClassifier` — ONNX-backed classifiers.
"""

from gazecontrol.gesture.classifier import GestureClassifier
from gazecontrol.gesture.feature_extractor import FeatureSet, GestureFeatureExtractor
from gazecontrol.gesture.hand_detector import HandDetector
from gazecontrol.gesture.mlp_classifier import MLPClassifier, TCNClassifier
from gazecontrol.gesture.rule_classifier import RuleClassifier

__all__ = [
    "FeatureSet",
    "GestureClassifier",
    "GestureFeatureExtractor",
    "HandDetector",
    "MLPClassifier",
    "RuleClassifier",
    "TCNClassifier",
]
