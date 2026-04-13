# GazeControl — Architecture

## Pipeline overview

```
Webcam
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  CaptureStage                                            │
│  FrameGrabber (threaded, auto-recovery)                  │
│    └─► FramePreprocessor (CLAHE LAB + sharpen + quality) │
│          └─► FrameContext { frame_bgr, frame_rgb, ok }   │
└─────────────────────────────────────────────────────────┘
  │  FrameContext
  ▼
┌─────────────────────────────────────────────────────────┐
│  GazeStage                                               │
│  MediaPipe FaceMesh ──► landmarks                        │
│  FaceCropper ──────────► crop_from_landmarks             │
│  TinyMLP (eyetrax) ──► raw_gaze (yaw, pitch)             │
│  L2CSNet ONNX (opt.) ──► l2cs_gaze                       │
│  Ensemble (weighted) ──► gaze_xy_screen                  │
│  OneEuroFilter ────────► gaze_filtered                   │
│  DriftCorrector ───────► gaze_corrected                  │
│  FixationDetector ─────► FixationEvent                   │
└─────────────────────────────────────────────────────────┘
  │  + gaze_corrected, fixation_event
  ▼
┌─────────────────────────────────────────────────────────┐
│  GestureStage                                            │
│  HandDetector (MediaPipe Tasks, VIDEO mode)              │
│    └─► FeatureSet (normalized landmarks + ratios)        │
│  RuleClassifier ──► gesture_label (fast path)            │
│  MLPClassifier  ──► gesture_label (slow path, ONNX)      │
└─────────────────────────────────────────────────────────┘
  │  + gesture: GestureLabel
  ▼
┌─────────────────────────────────────────────────────────┐
│  IntentStage                                             │
│  WindowSelector (hit-test, z-order cache)                │
│  IntentStateMachine (FSM)                                │
│    States: IDLE, HOVER, DWELL, DRAG, RESIZE, SCROLL      │
│    └─► Action(type=ActionType, target_hwnd, delta)       │
└─────────────────────────────────────────────────────────┘
  │  + action: Action | None
  ▼
┌─────────────────────────────────────────────────────────┐
│  WindowsManager (Win32 API)                              │
│  BaseWindowManager.execute(action) dispatches:           │
│    move / resize / close / minimize / scroll / focus     │
└─────────────────────────────────────────────────────────┘
  │  side-effects (window moved/resized/closed)
  ▼
┌─────────────────────────────────────────────────────────┐
│  OverlayWindow (PyQt6, signal-driven)                    │
│  HUDRenderer paints:                                     │
│    gaze dot / fixation ring / state label / perf stats   │
└─────────────────────────────────────────────────────────┘
```

## Threading model

```
Main thread (Qt event loop)
  │
  ├── QThread: PipelineThread
  │     runs Orchestrator._loop()
  │     emits frame_processed(FrameContext) via QueuedConnection
  │
  └── QThread: FrameGrabberThread
        runs FrameGrabber._capture_loop()
        writes to shared _frame_buffer under _cap_lock
```

## Calibration flow

```
gazecontrol --calibrate --profile <name>
  │
  ├── open_camera_dshow()         (DirectShow backend, exposure locked)
  ├── wait_for_face()             (MediaPipe FaceMesh, 25-pt grid)
  ├── collect_calibration_points()
  ├── GazeMapper.fit()            (Ridge regression, poly-2 features)
  └── GazeMapper.save()           (.npz + meta.json in profiles/<name>/)
```

## Configuration hierarchy

```
settings.toml (repo root, not committed)
  overridden by
GAZECONTROL_* environment variables
  both resolved by
pydantic-settings (src/gazecontrol/settings.py)
```

## Key design decisions

| Decision | Rationale |
|---|---|
| 1€ filter over EMA | Adaptive cutoff: smooth during fixations, responsive during saccades |
| I-VT fixation detection | Velocity threshold is simpler + more robust than dispersion for 30fps webcam |
| TinyMLP + Ridge ensemble | TinyMLP learns non-linearities, Ridge fills when L2CS model absent |
| ONNX for runtime inference | No PyTorch dependency at runtime; DirectML for GPU acceleration on Windows |
| `@dataclass FrameContext` | Typed, immutable-by-default; avoids dict key typos across stages |
| Enum for states/gestures/actions | Eliminates stringly-typed dispatch bugs; IDE auto-completion |
| npz persistence for GazeMapper | Version-stable; sklearn pickle breaks across minor versions |
