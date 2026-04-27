# ADR-0001 — Single-thread synchronous stage pipeline

Status: Accepted (v0.7.0). Reaffirmed in v0.8.0.

## Context

GazeControl runs a real-time loop at 30 fps with hard ordering constraints
(capture → gaze → gesture → fusion → interaction → action). Mediapipe's
`HandLandmarker` and ONNX Runtime sessions are not safe to share across
threads.

## Decision

Pipeline stages run sequentially in a single dedicated thread driven by
`PipelineEngine.run()`. The engine is synchronous and Qt-free. A thin
`QtPipelineThread` adapter bridges to the Qt event loop via `pyqtSignal`
with `QueuedConnection` so the GUI receives frames on the main thread.

## Consequences

- Predictable per-tick latency, no inter-stage queue back-pressure logic.
- Cannot exploit per-stage parallelism — acceptable because hand and gaze
  detection are CPU-bound and run < 33 ms/frame combined.
- Stop is cooperative: `request_stop()` is thread-safe (Event-backed).
