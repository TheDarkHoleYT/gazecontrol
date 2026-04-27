# GazeControl — Architecture

This document captures the runtime architecture, the per-mode data flow,
and the architectural decisions that shape the codebase.

---

## 1. Top-level overview

```
                      ┌─────────────────────────────┐
                      │  cli.main()                 │
                      │  ─ argparse + log setup     │
                      │  ─ ModeSelectorDialog       │
                      │  ─ PipelineFactory.build()  │
                      └──────────────┬──────────────┘
                                     │
              ┌──────────────────────┴────────────────────────┐
              ▼                                               ▼
        Mode A: HAND_ONLY                            Mode B: EYE_HAND
              │                                               │
              ▼                                               ▼
   ┌──────────────────┐                     ┌──────────────────────────────┐
   │ PipelineEngine   │                     │ PipelineEngine               │
   │ (sync loop, FPS) │                     │ (sync loop, FPS)             │
   └───────┬──────────┘                     └───────┬──────────────────────┘
           ▼                                          ▼
   Capture → Gesture → Interaction → Action     Capture → Gaze → Gesture
                                                     → PointerFusion
                                                     → Interaction → Action

         ╔════════════════════════════════════════════════════════════╗
         ║  QtPipelineThread emits frame_processed (Qt QueuedConn.)   ║
         ║   → OverlayWindow (HUD + LauncherPanel) on main thread     ║
         ╚════════════════════════════════════════════════════════════╝
```

---

## 2. Stage Protocol

Every stage implements the same interface:

```python
class PipelineStage(Protocol):
    name: str
    def start(self) -> bool: ...
    def stop(self) -> None: ...
    def process(self, ctx: FrameContext) -> FrameContext: ...
```

`FrameContext` is a single mutable dataclass passed through every stage
each tick. Each stage owns a slice of fields (capture, gaze, gesture,
fusion, interaction). The engine short-circuits the tick when
`capture_ok=False` to avoid wasting CPU on a missing frame.

---

## 3. Mode A — HAND_ONLY (zero regression)

```
[Camera]
   │ FrameGrabber (bg thread)
   ▼
CaptureStage
   │ frame_bgr / frame_rgb / quality / capture_ok
   ▼
GestureStage
   │ MediaPipe HandLandmarker
   │ → FeatureSet (21 landmarks → 30+ features)
   │ → RuleClassifier ◁ TCN/MLP fallback ◁ GestureFusion
   │ → FingertipMapper(landmark 8) → fingertip_screen
   │ → PinchTracker → pinch_event
   │ writes ctx.pointer_screen = ctx.fingertip_screen
   ▼
InteractionStage
   │ WindowHitTester(pointer_screen) → hovered_window
   │ InteractionFSM(pinch_event, pointer, scroll) → Interaction
   ▼
ActionStage
   └─ WindowsManager.move_window / resize_window / click_at / scroll_at
      AppLauncher  (TOGGLE_LAUNCHER → overlay_bridge)
      on_stop callback
```

In Mode A the gaze fields on `FrameContext` stay at their defaults
(`gaze_screen=None`, `pointer_source="hand"`), so the HUD and the FSM
behave exactly as in v0.6.

---

## 4. Mode B — EYE_HAND

```
[Camera]
   │
   ▼
CaptureStage
   │ frame_bgr, frame_rgb
   ▼
GazeStage  ────────────────────────────────────────────────┐
   │ GazeBackend.predict(frame_bgr, frame_rgb, t0)         │
   │   ├─ EyetraxBackend (landmarks → TinyMLP)             │
   │   ├─ L2CSBackend (face_crop → ONNX → GazeMapper)      │
   │   └─ EnsembleBackend (weighted blend)                  │
   │ OneEuroFilter (per-axis adaptive smoothing)           │
   │ DriftCorrector (edge snap + implicit recalibration)   │
   │ FixationDetector (I-VT classification)                │
   │ writes ctx.gaze_screen, ctx.gaze_event,                │
   │        ctx.gaze_confidence, ctx.face_present          │
   ▼                                                       │
GestureStage  (same as Mode A)                             │
   │ writes ctx.fingertip_screen + ctx.pointer_screen      │
   ▼                                                       │
PointerFusionStage  ◁───────────────────────────────────────┘
   │ priority:
   │   1. hand confidence ≥ τ_hand → fingertip wins
   │   2. fixation centroid (gaze conf ≥ τ_gaze)
   │   3. raw gaze (saccade / pursuit)
   │   4. low-conf hand fallback
   │ optional: gaze_assisted_click overrides 1
   │ writes ctx.pointer_screen + ctx.pointer_source
   ▼
InteractionStage   (reads pointer_screen, fallback fingertip_screen)
   ▼
ActionStage        (drift feedback → GazeStage.on_user_action when enabled)
```

---

## 5. Module boundaries

| Module                    | Responsibility                                           |
|---------------------------|----------------------------------------------------------|
| `cli.py`                  | argparse, dialog, mode resolution, Qt + thread bootstrap |
| `runtime/input_mode.py`   | InputMode enum, runtime.toml persistence                 |
| `runtime/pipeline_factory`| Stage list per mode + DI wiring                          |
| `pipeline/engine.py`      | Loop, FPS pacing, profiler, error policy                 |
| `pipeline/qt_adapter.py`  | QThread wrapper + thread-safe frame_processed signal     |
| `pipeline/context.py`     | Mutable FrameContext (single source of truth)            |
| `pipeline/{stages}.py`    | One file per stage; pure Python, framework-agnostic      |
| `gaze/backend.py`         | GazeBackend Protocol, GazePrediction value object        |
| `gaze/{backend}_*.py`     | Three concrete backends (eyetrax, l2cs, ensemble)        |
| `gesture/*`               | MediaPipe + rule/MLP/TCN + GestureFusion                 |
| `interaction/*`           | InteractionFSM + WindowHitTester + grip_region           |
| `filters/*`               | OneEuro, Kalman, DeadZone, AccelerationCurve             |
| `overlay/*`               | PyQt6 HUD + LauncherPanel + ModeSelectorDialog           |
| `calibration/runner.py`   | Qt 3×3 grid + GazeMapper.fit + persistence               |
| `window_manager/*`        | Win32 wrappers (move/resize/click/scroll)                |
| `settings.py`             | pydantic-settings nested groups + env vars + TOML        |
| `paths.py`                | platformdirs + cached path factory                       |
| `errors.py`               | GazeControlError hierarchy with user_message()           |
| `logging_config.py`       | Rotating file + JSON opt-in + run_id                     |

---

## 6. Configuration surface

`AppSettings` (pydantic-settings) composes:

| Group         | Highlights                                                                |
|---------------|---------------------------------------------------------------------------|
| `camera`      | index, width/height/fps, warmup, blur threshold, CLAHE                    |
| `gesture`     | min/max hands, confidence thresholds, pointer filters, sensitivity        |
| `interaction` | pinch hysteresis, tap_ms, hold_ms, grip_ratio, cooldown, min window size  |
| `gaze`        | backend (eyetrax/l2cs/ensemble), 1€ params, drift, blink hold, profile    |
| `fusion`      | hand/gaze thresholds, divergence_px, gaze_assisted_click                  |
| `runtime`     | input_mode, mode_selector_remember, last_chosen_mode, show_mode_selector  |
| `launcher`    | apps list, columns, opacity                                               |
| `overlay`     | pointer/grip/drag/resize/targeting colours                                |
| `logging`     | level, format (text/json), rotation_mb, profiler_log_every_n              |

Override via env (`GAZECONTROL_<GROUP>__<FIELD>=...`) or `settings.toml`.
The startup mode chosen via the dialog is persisted in
`<user_config>/gazecontrol/runtime.toml`.

---

## 7. Architectural decision records

### ADR-001 — Stage Protocol over inheritance
*Decision*: Pipeline stages use a `runtime_checkable` Protocol
(`PipelineStage`) instead of an abstract base class.
*Why*: Test doubles (mocks, simple dataclasses) satisfy the contract
without inheritance ceremony, and stages stay framework-agnostic.

### ADR-002 — Two backends per gaze input
*Decision*: Keep `EyetraxBackend` and `L2CSBackend` as separate
implementations under a common Protocol, with `EnsembleBackend` as a
weighted composer.
*Why*: Lets us start cheap (eyetrax landmarks, CPU-only) and degrade
gracefully when the L2CS ONNX model or GPU is missing. Ensemble keeps
both honest under shifting head pose / lighting.

### ADR-003 — Single FrameContext, mutated in place
*Decision*: Stages mutate the same `FrameContext` instance and return it.
*Why*: Profiled allocation in Python; immutable contexts cost ~80 µs /
frame on copies. Field ownership documented per stage prevents data
races (single thread + single mutator per field).

### ADR-004 — Hand has cursor priority in Mode B
*Decision*: When both hand and gaze are valid, hand drives the pointer.
*Why*: Hand has sub-pixel precision and explicit user intent; gaze
suffers ~30–80 px residual error even after calibration. Gaze becomes
the *target selector* (hover, distant windows), not the cursor.

### ADR-005 — Calibration uses Qt, not OpenCV imshow
*Decision*: The calibration runner is a `QWidget`, not a `cv2.imshow`
fullscreen.
*Why*: The legacy v0.3 OpenCV calibration conflicted with the PyQt6
overlay (two competing fullscreen surfaces) and could not be themed.
Qt also gives us animated targets and DPI-correct positioning.

### ADR-006 — Mode selector at startup, not in-app toggle
*Decision*: The mode is fixed at startup and persists via
`runtime.toml`. Switching requires re-launching.
*Why*: Switching modes mid-run would require re-instantiating the
pipeline, retraining filters, reloading models — all ~1 s of stutter.
A startup dialog is simpler and matches user mental model
("am I in eye mode today?").

### ADR-007 — Optional eyetrax + sklearn via `[eye]` extra
*Decision*: `eyetrax` and `scikit-learn` ship as optional dependencies.
*Why*: Hand-only users (the majority on launch) should not need to
install ~100 MB of ML stack just to pinch.

### ADR-008 — L2CS model not bundled
*Decision*: The 100 MB L2CS-Net ONNX is downloaded on demand; it does
not live in the wheel.
*Why*: PyPI size limits + most users will never enable Mode B.

---

## 8. Threading model

| Thread             | Owner            | Notes                                                   |
|--------------------|------------------|---------------------------------------------------------|
| Main (Qt event)    | `QApplication`   | Owns `OverlayWindow`, `ModeSelectorDialog`              |
| Pipeline           | `QtPipelineThread` (`QThread`) | Runs `PipelineEngine.run()`               |
| Capture            | `FrameGrabber`   | Reads from OpenCV; produces a single-buffer queue       |
| ONNX worker (DML)  | onnxruntime      | Owned by `L2CSModel`; called only from pipeline thread  |

`overlay.update(...)` is thread-safe: it emits a Qt `QueuedConnection`
signal that hops to the main thread automatically.

---

## 9. Future extensibility

- **More backends**: implement `GazeBackend` Protocol (e.g. WebGazer.js
  via subprocess, dedicated Tobii / Pupil-Labs hardware). Register in
  `PipelineFactory._build_gaze_backend()`.
- **Voice input**: another `PipelineStage` between capture and
  interaction, writing into `FrameContext` (e.g. `voice_command`).
- **Mode C — Eye Only**: add `InputMode.EYE_ONLY`, drop `GestureStage`,
  introduce a dwell-based click in `PointerFusionStage` or a new
  `DwellClickStage`.
- **Calibration dataset capture**: persist raw (yaw, pitch, target_x,
  target_y) to `<profiles>/<name>.dataset.npz` so users can retrain
  offline with a richer regressor.

---

## 10. Testing layout

| Layer        | Where                              | Tooling                |
|--------------|------------------------------------|------------------------|
| Unit         | `tests/<module>/test_*.py`         | pytest + hypothesis    |
| Integration  | `tests/runtime/test_pipeline_*.py` | pytest + pytest-qt     |
| Coverage     | `coverage.xml`, fail_under=80      | pytest-cov             |
| Type-check   | `mypy --strict` on listed modules  | mypy                   |
| Lint         | ruff (selected rules)              | ruff + pre-commit      |

Stage tests use stub backends (`tests/gaze/test_backend_protocol.py`,
`tests/pipeline/test_gaze_stage.py`) so the suite runs without ONNX,
eyetrax, or a webcam.
