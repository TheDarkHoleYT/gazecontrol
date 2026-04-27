# ADR-0006 — Public API stability contract

Status: Accepted (v0.8.0).

## Context

Earlier versions exposed internals via re-export. Downstream packages
were coupled to `gazecontrol.gesture.mlp_classifier.MLPClassifier`
even though that symbol moved between releases.

## Decision

The public surface is exactly what each subpackage's `__all__`
enumerates, plus the top-level `__version__`. Concretely:

| Surface             | Stable                                                          |
|---------------------|-----------------------------------------------------------------|
| `gazecontrol`       | `__version__`, `AppSettings`, `get_settings`, `InputMode`        |
| `gazecontrol.pipeline` | `PipelineEngine`, `PipelineStage`, `FrameContext`, `QtPipelineThread` |
| `gazecontrol.gaze`  | `GazeBackend` (Protocol), `GazePrediction`, `GazeMapper`, `FaceCropper`, `FixationDetector`, `GazeEvent`, `DriftCorrector` |
| `gazecontrol.gesture` | `GestureClassifier`, `FeatureSet`, `GestureFeatureExtractor`, `HandDetector`, `RuleClassifier`, `MLPClassifier`, `TCNClassifier` |
| `gazecontrol.runtime` | `InputMode`, `load_persisted_mode`, `persist_mode`              |

Anything else is internal and may move, rename, or disappear in any
release. SemVer applies only to the table above.

Backwards-compatible aliases (e.g. `GazeControlPipeline`) live for one
minor release before removal.

## Consequences

- Breaking the table requires a major version bump or a documented
  deprecation cycle (one minor with `DeprecationWarning`).
- Internal refactors no longer invalidate downstream code.
- A snapshot test (`tests/api/test_public_surface.py`) guards every
  `__all__` against accidental exposure.
