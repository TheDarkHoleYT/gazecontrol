# Changelog

All notable changes to GazeControl are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.8.0] — 2026-04-27 (enterprise refactor)

### Added

- **Public API contract** — every subpackage now declares an explicit
  `__all__`; `tests/api/test_public_surface.py` snapshot-tests it. See
  [ADR-0006](docs/adr/0006-public-api-contract.md).
- **`gazecontrol/py.typed` marker** (PEP 561). Wheels carry inline type
  information so downstream `mypy` runs see GazeControl's signatures.
- **Crash handler stack** — `gazecontrol.runtime.crash.install_crash_handlers`
  registers `sys.excepthook`, `threading.excepthook`, `faulthandler`, Qt
  message handler, and SIGTERM/SIGBREAK → graceful-shutdown bridges.
  See [ADR-0005](docs/adr/0005-crash-handler-stack.md).
- **CLI exit codes** — `errors.exit_code_for(exc)` maps each
  `GazeControlError` subtype to a stable exit code (10–15). The CLI now
  uses these instead of always exiting with 1/2.
- **`ModelDownloadError`** — subclass of `ModelLoadError` for retry-exhausted
  downloads.
- **`utils/threading_helpers.py`** — `ShutdownToken` and
  `run_with_timeout()` for bounded worker calls.
- **`utils/correlation.py`** — `frame_id` propagation via `contextvars`
  + a `CorrelationFilter` that injects it into every `LogRecord`.
- **Profiler counters / gauges** — `frames_total`, `frames_dropped_total`,
  `stage_errors_total{stage=...}`, `camera_restarts_total`,
  `model_load_failures_total`, `pipeline_actual_fps`, `gaze_confidence`,
  `hand_confidence`. Exported alongside the existing latency metrics.
- **CLI ops surface** — `--healthcheck` (exit code matches failure class),
  `--doctor --json` (machine-readable), `--dump-config --resolved` (also
  emits resolved env vars + paths), `--log-modules MOD:LEVEL,...`
  (per-module level overrides), `GAZECONTROL_LOG_LEVEL` env var.
- **Architecture Decision Records** — `docs/adr/0001`–`0006` covering
  pipeline, settings precedence, model pinning, mypy stance, crash stack,
  and the public-API contract.
- **Supply-chain CI** — `pip-audit --strict`, `bandit --severity-level
  medium`, CycloneDX SBOM artefact, wheel + sdist build with `twine check`.
- **Release automation** — `.github/workflows/release.yml` builds, signs
  with Sigstore, publishes to TestPyPI on `v*-rc*` tags and PyPI on `v*`
  via Trusted Publishing, and creates a GitHub Release.
- **CodeQL** — `.github/workflows/codeql.yml` runs the security-and-quality
  query suite weekly + on every push/PR.
- **Dependabot** — weekly updates for `pip` and `github-actions` ecosystems.

### Changed (BREAKING)

- **mypy is strict everywhere.** The 29 `[tool.mypy.overrides]` entries
  are gone; the codebase compiles cleanly under `strict = true`. Internal
  refactors, signatures tightened, untyped `# type: ignore` comments
  replaced with specific error codes. See
  [ADR-0004](docs/adr/0004-mypy-strict-no-overrides.md).
- **`FrameContext` exposes `frame_id`** (monotonic per-tick counter) and
  no longer carries hidden `_HandResult` aliasing.
- **`HudData` TypedDict** replaces the implicit dict shape consumed by
  `HUDRenderer.render`.
- **`HudState` is `frozen=True`** — callers build a fresh snapshot per
  tick instead of mutating in place.
- **PyQt6 is a hard dependency** at the type level; the legacy
  `class _OverlayWidget(QWidget if HAS_PYQT else object)` shim is gone.
  `HAS_PYQT = True` is retained for backward-compat with test fixtures.
- **`FrameGrabber.read_bgr()`** returns a read-only ndarray view (no copy).
  Consumers must copy before mutating; cuts ~10–30 % GC pressure at 30 fps.
- **`FrameGrabber._capture_loop`** no longer holds `_cap_lock` across
  `cap.read()` — a stalled camera no longer blocks `stop()` for seconds.
  This reverses the v0.7.1 lock-during-read fix in favour of a snapshot
  pattern that handles `_restart_camera()` raceless via the cv2 sentinel.
- **`QtPipelineThread`** uses `requestInterruption()` / `isInterruptionRequested()`
  instead of an ad-hoc `_stopping` Event — eliminates the race that could
  emit a frame after `stop()` returned.
- **`InteractionFSM.update()`** is wrapped in a re-entrant lock to make
  state mutations atomic across the rare re-entrant callback path.
- **Specific exception classes** replace bare `except Exception` in hot
  paths (`gaze/*`, `gesture/*`, `pipeline/*`, `cli.py`, `logging_config.py`).
- **`pyproject.toml`** uses dynamic version sourced from
  `gazecontrol.__version__` — single source of truth.
- **`Development Status :: 4 - Beta`** classifier (was `3 - Alpha`).

### Fixed

- 312 mypy errors across 37 files when `strict = true` is enabled.
- 124 ruff findings (per-file ignores narrowed instead of broadened).
- `_extract_probas` ZipMap branch flattened to remove SIM102 nested-if.
- `interaction_fsm.update()` two-finger-scroll branch un-nested for clarity.
- `cli.py` `_detect_virtual_desktop` catches specific Win32 errors instead
  of bare `Exception`.
- `_cmd_doctor` JSON path keeps the same exit-code contract as the table.

### Roadmap deferred

- Config hot-reload via file watcher.
- OpenTelemetry traces (desktop app — not yet justified).
- HTTP `/health` endpoint (use `--healthcheck` instead).

---

## [0.7.1] — 2026-04-27 (audit hardening)

### Fixed

- **Gaze drift edge-snapping had its sign inverted** (`gaze/drift_corrector.py`).
  When the corrected gaze overshot a screen edge, the offset was updated in
  the wrong direction, pushing it further off-screen instead of converging
  back. Sign now mirrors the (already correct) implicit-recalibration path.
  Regression covered by two new convergence tests.
- **`GazeMapper.save` is now atomic.** The `.npz` and `.meta.json` files are
  staged under `.part` filenames and `os.replace`-d into place only after
  both writes succeed. A crash mid-save no longer destroys an existing
  calibration profile.
- **`runtime/input_mode.persist_mode` is now atomic.** `runtime.toml` is
  written to `runtime.toml.tmp` then renamed; failures leave the previous
  file intact.
- **`WindowSelector` hit-test uses half-open intervals.** Right and bottom
  edges no longer double-claim the boundary pixel between adjacent windows.
- **`FrameGrabber` capture loop holds `_cap_lock` across `cap.read()`** so a
  concurrent restart cannot release the `VideoCapture` underneath a reader.
  Consumers continue to use the separate `_frame_lock` and are not blocked.
- **`MLPClassifier.load` validates ONNX input dimension** against
  `FEATURE_ORDER` and refuses to load on mismatch — catches a silently
  retrained model before it produces garbage predictions.

### Security

- **MediaPipe model SHA256 checksums are now pinned** for `face_landmarker.task`,
  `hand_landmarker.task`, and `blaze_face_short_range.tflite`. New regression
  test asserts no entry in `_MODELS` ships unpinned. The
  `GAZECONTROL_ALLOW_UNPINNED_MODELS=1` escape hatch is preserved for
  development-only use against new upstream releases.

### Documentation

- `FeatureSet.to_vector` docstring corrected: it produces the 17-feature
  vector consumed by the **TCN** classifier; the MLP uses the 16-feature
  `FEATURE_ORDER` (no `thumb_dir_y`). Mismatch is now also enforced at
  runtime by the MLP shape guard above.
- `GestureFusion` module docstring clarifies that "Path 3" intentionally
  passes the rule label through with its below-threshold confidence (the
  existing tests already enforce this).

### Tests

- 12 new regression tests across `tests/gaze/test_drift_corrector.py`,
  `tests/gaze/test_gaze_mapper.py`, `tests/runtime/test_input_mode.py`,
  `tests/utils/test_model_downloader.py`, `tests/gesture/test_mlp_classifier.py`,
  and `tests/interaction/test_window_selector.py`. Total suite: **357
  passing**, coverage 83 %.

---

## [0.7.0] — 2026-04-27

### Added

#### Eye tracking — second input mode
- `InputMode` enum (`HAND_ONLY`, `EYE_HAND`) and runtime persistence in
  `<user_config>/gazecontrol/runtime.toml`.
- `RuntimeSettings`, `GazeSettings`, `FusionSettings`, `DriftCorrectorSettings`
  pydantic-settings groups (env-var overridable via `GAZECONTROL_*`).
- `GazeBackend` Protocol + `GazePrediction` value object.
- Three concrete backends: `EyetraxBackend` (landmark-based, optional
  `[eye]` extra), `L2CSBackend` (ONNX appearance-based, optional model
  download), `EnsembleBackend` (weighted blend, fallback-tolerant).
- `gaze/gaze_mapper.py`, `gaze/face_crop.py`, `gaze/fixation_detector.py`,
  `gaze/drift_corrector.py`, `gaze/l2cs_model.py` (recovered/cleaned from
  v0.3, made dependency-optional).

#### Pipeline integration
- `pipeline/gaze_stage.py` — backend wrapper with 1€ filter, drift
  corrector, I-VT fixation detector.
- `pipeline/pointer_fusion_stage.py` — priority matrix combining hand
  and gaze, with optional `gaze_assisted_click`.
- New `FrameContext` fields: `gaze_screen`, `gaze_event`, `gaze_blink`,
  `gaze_yaw_pitch_deg`, `face_present`, `gaze_confidence`,
  `pointer_screen`, `pointer_source`.
- `runtime/pipeline_factory.py` — assembles the per-mode stage list and
  shared callbacks. Mode A is bit-identical to v0.6; Mode B inserts
  gaze + fusion stages.

#### UI / CLI
- `overlay/mode_selector_dialog.py` — startup `QDialog` with two large
  cards (Hand-Only / Eye+Hand), "remember choice" checkbox.
- HUD additions: gaze ring + dot, mode badge, pointer-source hint.
- `--mode {hand,eye-hand}` and `--no-mode-selector` CLI flags.
- `--calibrate-gaze` runs the new Qt-based 3×3 calibration runner.
- `--doctor` now probes eyetrax availability, L2CS model, and gaze
  profile presence.

#### Tests
- 60+ new unit tests covering the gaze backends, fixation detector,
  drift corrector, gaze mapper, gaze stage, pointer fusion stage,
  runtime persistence, settings, and pipeline factory.
- Total suite: **345 tests passing**, coverage ~83 %.

### Changed

- `InteractionStage.process` reads `ctx.pointer_screen` (falls back to
  `ctx.fingertip_screen` for compatibility) so the same FSM serves both
  modes.
- `GestureStage.process` writes `ctx.pointer_screen` so HAND_ONLY mode
  bypasses fusion.
- `OverlayWindow.update` and `HudState` accept gaze + pointer-source +
  input-mode fields.
- `pyproject.toml`: version bumped to **0.7.0**, new `[eye]` optional
  extra (`eyetrax`, `scikit-learn`).

### Documentation

- `README.md` rewritten — quick start, mode comparison, configuration,
  troubleshooting, examples.
- `docs/architecture.md` rewritten — module map, per-mode data flow,
  ADRs 1–8, threading model, extensibility hooks.

---

## [0.6.0] — 2026-04-14

### BREAKING

- `FrameGrabber.read()` and `FrameGrabber._preprocess()` removed — use
  `read_bgr()` and let `CaptureStage` handle the flip/RGB conversion.
- `FingertipMapper.map()` now accepts `dt` and `confidence` keyword arguments.
  Callers that relied on the positional-only 2-arg signature still work
  (both args are optional).
- `GestureLabelsSettings.labels` default now includes all 9 canonical labels
  from `GestureLabel` (was 4 labels previously).
- `OverlayWindow.update()` parameter `frame_thumb` renamed to `frame_bgr`
  (pass full-resolution BGR frame; the renderer resizes internally).

### Added

#### Pointer filter stack (`gazecontrol.filters`)
- `OneEuroFilter` — adaptive low-pass filter (Casiez et al. 2012).
  Smooth at rest, responsive during fast motion.
- `KalmanFilter2D` — constant-velocity Kalman for 2-D screen coordinates.
  Compensates pipeline latency; measurement noise scaled by MediaPipe confidence.
- `AccelerationCurve` — smooth velocity-to-gain mapping (slow = precision
  dampened, fast = amplified).
- `DeadZone` — circular dead-zone with hysteresis; suppresses micro-jitter
  when the hand is stationary.

#### Settings (`gazecontrol.settings`)
- `PointerFilterSettings` (nested under `GestureSettings.pointer`) with
  sub-models `OneEuroSettings`, `KalmanPointerSettings`,
  `AccelerationCurveSettings`, `DeadZoneSettings`.  All filters individually
  enabled/disabled via env vars
  (`GAZECONTROL_GESTURE__POINTER__ONE_EURO__BETA`, etc.).
- `CameraSettings.enhance` flag (default `False`) — skip CLAHE+sharpening
  to save ~3–5 ms/frame when lighting is adequate.

#### Gesture labels (`gazecontrol.gesture.labels`)
- `GestureLabel` (`StrEnum`) — canonical gesture vocabulary shared by
  `RuleClassifier`, `MLPClassifier`, and `InteractionFSM`.
- `DEFAULT_LABELS` — ordered list of all 9 labels.

#### Profiler (`gazecontrol.utils.profiler`)
- `PipelineProfiler.percentiles()` — returns p50/p95/mean per stage.
- `PipelineProfiler.emit_prometheus(path)` — writes Prometheus text-format
  metrics file for node_exporter or manual inspection.
- Budget warning: logs a `WARNING` after 5 consecutive frames that exceed
  the target pipeline latency budget.

#### CLI (`gazecontrol.cli`)
- `--benchmark [SECONDS]` — runs the pipeline headless for N seconds (default
  30) and prints per-stage latency percentiles (p50 / p95 / mean).

#### Gesture ML v2 (`gazecontrol.gesture`)
- `TCNClassifier` (`mlp_classifier.py`) — temporal 1D-TCN classifier with an
  internal 30-frame sliding window.  Accepts single-frame `FeatureSet` input;
  buffers frames and runs ONNX inference when the window is full.  Labels read
  from `models/manifest.json` (written by `train_gesture_tcn.py`).
- `GestureFusion` — confidence-weighted rule-first + ML-fallback classifier.
  Rule ≥ 0.80 → rule; else ML ≥ 0.70 AND label ∈ FSM vocabulary → ML; else
  rule passthrough or `None`.
- `FeatureSet.to_vector()` — canonical 17-float vector for ML inference
  (5 finger states + 5 angles + palm_direction + vx + vy + thumb_index_dist
  + wrist_x + wrist_y + thumb_dir_y).
- `Paths.gesture_tcn_model()` — returns `models/gesture_tcn_v1.onnx` path.

#### Training tools (`tools/`)
- `tools/record_gesture_dataset.py` — interactive dataset recorder.  Opens
  webcam with countdown, saves 30-frame `[T, F]` windows to `.npz` files and
  updates `data/gestures/manifest.json`.
- `tools/train_gesture_tcn.py` — trains a 1D-TCN (3 dilation blocks, opset 17
  ONNX export), writes `models/training_report.md` (confusion matrix + per-class
  F1) and updates `models/manifest.json` with SHA256.

### Changed
- `FingertipMapper` filter pipeline: `raw → One-Euro → Kalman → dead-zone →
  acceleration curve → sensitivity scaling`.  Backward-compatible: passing
  `filter_cfg=None` (default) restores the original raw sensitivity-only
  behaviour.
- `HUDRenderer`: all `QColor`/`QFont`/`QPen`/`QBrush` objects pre-allocated
  in `__init__` (eliminates ~40 GC-eligible allocations per paint cycle).
  Camera thumbnail now uses `QImage.Format_BGR888` (zero-copy on PyQt6 ≥ 6.5)
  and is resized inside the renderer instead of on the pipeline thread.
- `pipeline/__init__.py` docstring updated to reflect actual stage order
  (CaptureStage → GestureStage → InteractionStage → ActionStage).
- `FramePreprocessor` docstring rewritten in English, removing stale
  eye-tracking / iris references.
- CI `--cov-fail-under` raised from 70 → 80 (matches local `pyproject.toml`).
- CI: Codecov upload step added; `security` job with `pip-audit` + `bandit`.

### Dependencies
- `onnxruntime>=1.17.0` added as a declared runtime dependency (was implicit).
- `[gpu]` optional extra added: `onnxruntime-gpu>=1.17.0`.
- `pytest-xdist>=3.5.0` added to `[dev]` for parallel test execution.

---

## [0.5.0] — 2026-04-13

### BREAKING — complete architecture pivot: hand-only control

Eye tracking (eyetrax, L2CS, calibration, drift correction, fixation detection) has been
removed in its entirety.  The system now controls the desktop exclusively via hand gestures
detected by MediaPipe.

#### Removed
- `gazecontrol.gaze` package (eyetrax backend, L2CS backend, ensemble, GazeMapper, BlazeFace
  face-presence, fixation detector, drift corrector, 1€ filter, compat shim).
- `gazecontrol.pipeline.gaze_stage` and `pipeline.calibration`.
- `gazecontrol.intent` package (replaced by `gazecontrol.interaction`).
- CLI flags `--calibrate`, `--adaptive`, `--calibrate-l2cs`, `--profile`.
- Settings groups `GazeSettings`, `FixationSettings`, `DriftSettings`.
- `OverlaySettings.gaze_dot_*` fields.
- `FrameContext` gaze fields (`gaze_raw`, `gaze_filtered`, `gaze_corrected`, `gaze_point`,
  `fixation_event`, `blink`, `gaze_status`, `landmarks`, `frame_bgr_raw`).
- Dependencies `eyetrax`, `onnxruntime-directml`, `scikit-learn`.

#### Added
- **`gesture.FingertipMapper`** — normalized MediaPipe landmark → virtual-desktop pixel
  (linear, full-frame → full virtual desktop, zero calibration required).
- **`gesture.PinchTracker`** — hysteresis-based pinch event emitter (DOWN/HOLD/UP).
- **`interaction` package** — enterprise FSM + helpers:
  - `InteractionFSM` — IDLE → PINCH_PENDING → TAP/DRAG/RESIZE → COOLDOWN.
  - `GripRegion.is_in_resize_grip()` — detects bottom-right resize corner.
  - `WindowHitTester` — point-in-window lookup (wraps existing Win32 EnumWindows).
  - `HoveredWindow`, `Interaction`, `InteractionKind` value objects.
- **`window_manager.AppLauncher`** — subprocess-based detached app launcher.
- **`window_manager.WindowsManager`** — new primitives:
  - `click_at(x, y)` — synthetic `SetCursorPos` + `SendInput` left click.
  - `double_click_at(x, y)` — two synthetic clicks.
  - `scroll_at(x, y, delta)` — positioned `SendInput` wheel.
  - `get_window_rect(hwnd)` — Win32 rect helper.
- **`overlay.LauncherPanel`** — semi-transparent Qt grid panel for app launching.
- **`overlay.HudState`** — rewritten for hand-only fields.
- **`pipeline.InteractionStage`** — drives `InteractionFSM`, resolves hovered window.
- **`pipeline.ActionStage`** — rewritten dispatcher: CLICK/DRAG/RESIZE/SCROLL/TOGGLE_LAUNCHER.
- **`settings.InteractionSettings`** — all timing and geometry thresholds tunable via env.
- **`settings.LauncherSettings`** — configurable app list (name, exe, args, icon).

#### Changed
- Pipeline: `CaptureStage → GestureStage → InteractionStage → ActionStage` (4 stages).
- `GestureStage` now produces `fingertip_screen`, `pinch_event`, `two_finger_scroll_delta`.
- `OverlayWindow.update()` now accepts `HudState` with hand-only fields.
- `HUDRenderer` now draws: fingertip cursor dot, window outline, resize-grip hint.
- `Paths.launcher_config()` added; `Paths.l2cs_model/face_detector/face_landmarker` removed.
- `pyproject.toml` version `0.5.0`; removed `eyetrax`, `onnxruntime-directml`, `scikit-learn`.
- `fail_under` kept at 80; actual coverage **82.95 %** (219 tests, all passing).

#### Interaction model (new)
| Gesture | Action |
|---|---|
| Index finger visible | Pointer cursor on screen (linear full-frame mapping) |
| Quick pinch (< 220 ms) | Left click at fingertip |
| Double pinch (< 420 ms apart) | Toggle app launcher |
| Pinch-hold (> 280 ms) on window body | Drag window |
| Pinch-hold on bottom-right corner (18 %) | Resize window |
| Index + middle extended, vertical movement | Scroll on window under pointer |

---

## [0.4.2] — 2026-04-13

### Fixed (gaze — critical)
- **`L2CSBackend.is_ready()` wrong attribute** — `getattr(mapper, "_fitted", False)` read
  a private non-existent attribute; corrected to use the public `is_fitted` property. This
  was the root cause of `gaze=0.0ms` / `gaze_status="no_backend"` every tick in L2CS mode.
- **`GazeStage.start()` ensemble branch left `estimator=None`** — on `ProfileNotFoundError`
  the ensemble branch returned without assigning `self.estimator`, guaranteeing the
  `no_backend` short-circuit forever. Now falls back to L2CS-only.
- **`GazeMapper` never loaded in L2CS mode** — `start()` now loads
  `{profile_name}.mapper.npz` via `GazeMapper.load()` for l2cs and ensemble modes.
- **Default `model_mode` changed from `"mlp"` to `"l2cs"`** — L2CS requires no per-session
  eyetrax calibration and works immediately after `--calibrate-l2cs`.
- **Distinguishes `uncalibrated` from `no_backend`** — `gaze_status="uncalibrated"` is now
  set when the backend is present but the mapper is not fitted, giving the HUD a yellow dot
  with "CALIBRATE" label instead of the orange "NO CAL".

### Added
- **`gazecontrol --calibrate-l2cs`** — new 9-point L2CS calibration routine that collects
  yaw/pitch samples from L2CS-Net, fits `GazeMapper`, and saves `{profile}.mapper.npz`.
- **`gazecontrol --version`** — prints version and exits.
- **`gazecontrol --dump-config`** — prints resolved `AppSettings` as JSON and exits.
- **`gazecontrol --doctor`** — probes camera, L2CS ONNX model, BlazeFace model, ORT
  providers, GazeMapper profile, and PyQt6. Prints a ✓/✗ table and exits non-zero on
  any failure. Primary diagnostic tool for "gaze doesn't work" reports.
- **Startup banner** — single INFO log line with version, run_id, profile, mode, Python,
  OS, log file path, Qt availability, and ORT providers.
- **`run_id`** — 8-char hex injected into every log line (`[abc12345]`) for cross-session
  correlation. Visible in both text and JSON log formats.
- **`%(threadName)s`** added to default log format.
- **`user_message()`** on all `GazeControlError` subclasses — returns an actionable
  troubleshooting string shown by the CLI on fatal errors.
- **`TargetWindow` / `Action` dataclasses** in `FrameContext` — replace `dict[str, Any]`.
- **`HudState` dataclass** in `overlay/hud_state.py` — typed HUD snapshot; `OverlayWindow.update`
  accepts either `hud_state=` or legacy keyword arguments.
- **`LoggingSettings.profiler_log_every_n`** — configurable via env/TOML
  (`GAZECONTROL_LOGGING__PROFILER_LOG_EVERY_N`).

### Improved
- **Engine busy-loop** — on `capture_ok=False` the engine now sleeps the remaining
  frame budget instead of a flat 5 ms, preventing ~200 Hz spin on camera drop.
- **Stage-name hardcode removed** — engine skip logic uses `skip_on_capture_fail`
  stage attribute instead of `stage.name == "capture"`.
- **QThread shutdown** — `QtPipelineThread.stop()` calls `terminate()` + `wait(1000)`
  if the thread does not exit within 3 s, preventing process hang.
- **Overlay shutdown order** — `overlay.stop()` is called before `qt_thread.stop()` so
  queued Qt events drain on the live event loop.
- **SIGTERM / SIGBREAK handlers** installed on Windows so Ctrl-Break exits cleanly.
- **Engine `set_on_frame` / `set_on_shutdown`** — RuntimeError guard removed; simple
  assignment. Prevents `PipelineEngine` construction reuse failures.
- **L2CS ORT provider logged** at load time (`providers=...`).
- **mypy strict** extended to `pipeline/context`, `pipeline/stage`, `pipeline/engine`,
  `pipeline/action_stage`, `pipeline/intent_stage`, `pipeline/gaze_stage`, `settings`,
  `paths`, `overlay/hud_state`.

### Removed
- `utils/clock.py` — unused `Clock` / `MonotonicClock` / `FakeClock` abstractions.
- `PipelineStageError.__reduce__` — pickling support with no consumer.
- Duplicate `_PreprocessingCapture` method overrides (`isOpened`, `release`, `set`, `get`)
  — already handled by `__getattr__`.
- Stale docstring references to deleted `main.py` and `orchestrator.py`.

---

## [0.4.1] — 2026-04-13

### Fixed
- **Pipeline engine lifecycle** — `stop_stages()` was double-reversing the stage list
  via ExitStack, producing forward-order teardown. Rewritten as a plain reversed loop.
  `run()` now fires `on_shutdown` even when `start_stages()` fails (resource leaks fixed).
  Capture-stage exceptions force `ctx.capture_ok = False` before the short-circuit.
- **Qt adapter private-API misuse** — `QtPipelineThread` no longer mutates private
  `_on_frame` / `_on_shutdown` attributes; uses new public `set_on_frame` / `set_on_shutdown`
  setters. Added `_stopping` event to prevent cross-thread signal emission after stop.
- **GazeStage uncalibrated start** — a missing calibration profile (`ProfileNotFoundError`)
  no longer aborts the pipeline; the stage continues in uncalibrated mode and returns `True`.
  Brittle `"not found" in str(exc)` sniffing replaced with typed `ProfileNotFoundError`.
- **GazeStage resource leak** — `stop()` now calls `backend.close()` and clears the reference.
- **Calibration resource leaks** — `_try_open_candidate` uses `try/finally` so
  `VideoCapture` is always released on error paths. `wait_for_face` destroys the
  OpenCV "Calibration" window in `finally`. `assert quality is not None` replaced
  with `if quality is None: continue`.
- **CLI SIGINT** — restored `signal.default_int_handler` instead of `SIG_DFL` so
  `finally` blocks and `except KeyboardInterrupt` run on Ctrl-C. Return-type
  annotation corrected to 4-tuple.
- **L2CS loader** — non-`FileNotFoundError` ORT failures now re-raised as
  `ModelLoadError` instead of being silently swallowed. Logs model filename only.
- **GazeMapper NaN safety** — zero-variance scaler columns clamped to 1.0 to
  prevent `inf` divisions. `predict()` returns `None` for non-finite yaw/pitch.
- **DriftCorrector stale return** — `correct()` now recomputes `(cx, cy)` after
  `_update_edge_snapping` so the output reflects the updated offset (not the
  pre-snap value from the previous tick).
- **One-Euro filter reset** — `reset()` restores `_freq` to the initial frequency;
  `alpha()` clamps frequency to ≥ 1e-6 to prevent division-by-zero.
- **FacePresence half-constructed init** — MediaPipe detector creation wrapped in
  `try/except`; `detect()` guards against `None` detector; `__del__` safety net added.
- **Model downloader security** — every unverified (SHA256=None) model download is
  refused unless `GAZECONTROL_ALLOW_UNPINNED_MODELS=1`. HTTPS-only enforcement.
  Single scalar timeout (was indexing a tuple). Logs filename only.
- **MLP classifier output parsing** — introspects ONNX output metadata at load time;
  supports both ZipMap and raw-ndarray probability outputs. `FEATURE_ORDER` tuple
  drives vectorization to match training data order (now includes `wrist_x`/`wrist_y`).
- **Window manager Win32** — `move_window` and `resize_window` use `SetWindowPos`
  with `SWP_NOZORDER|SWP_NOACTIVATE` (no repaint flicker). `bring_to_front` uses
  `AttachThreadInput` fallback and logs a warning when `SetForegroundWindow` fails.
  `scroll` replaced with `SendInput(MOUSEEVENTF_WHEEL)` — works in Chromium/Electron.
- **Window selector filtering** — cloaked UWP windows (`DWMWA_CLOAKED`), tool
  windows (`WS_EX_TOOLWINDOW`), and owned/child windows are excluded from hits.
  `invalidate()` method added for force-refresh on state transitions. Coordinate
  space documented.
- **State machine timer bleed** — `_transition_to()` unconditionally resets all
  per-state fields; `RELEASE` in READY transitions to IDLE instead of being dropped;
  target-window dict deep-copied on READY→ACTING.
- **Gesture feature extractor velocity** — centroid buffer now stores `(ts, cx, cy)`
  tuples; velocity computed over real wall-clock time, preventing halved measurements
  on dropped frames.
- **Rule classifier swipe threshold** — removed erroneous `/100` divisor; threshold
  is now applied directly in px/s. `CLOSE_SIGN` detection uses `thumb_dir_y`
  (lm[4].y − lm[2].y) instead of the unreliable `wrist_y > 0.3` heuristic.
- **GestureStage allocation** — heavy resources (MediaPipe HandDetector, MLP ONNX)
  deferred from `__init__` to `start()`; `stop()` now closes both detector and
  classifier; accepts optional `settings` argument.
- **Overlay thread safety** — `_widget` reads/writes protected by `threading.Lock`;
  `deleteLater()` scheduled after `invokeMethod(close)`. `hud_renderer.py` uses
  `time.monotonic()` instead of `time.time()`.
- **Profiler tick count** — `_tick_count` increment and modulo check now performed
  under `_lock` to prevent lost updates under concurrent ticks.
- **DPI awareness** — CLI `_detect_screen()` now calls
  `SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)`
  (Windows 10 1607+) for correct rects on mixed-DPI multi-monitor setups, with
  automatic fallback to `SetProcessDPIAware` on older OS versions.
- **Settings** — ensemble-weight validator raises `ValueError` when both weights are
  0 instead of silently resetting. Duplicate `log_level` field removed. `get_settings()`
  uses double-checked locking. `configure_logging()` removes stale root handlers
  before applying new config.
- **eyetrax compat patches** — `wait_for_face_and_countdown` patch now covers
  `eyetrax.calibration.adaptive` module. Snapshots taken before touching any module
  so rollback is always possible; `revert_patches()` exposed publicly.
- **FrameContext annotations** — `target_window` corrected from `int | None` to
  `dict[str, Any] | None`; `action` corrected from `Action | None` to
  `dict[str, Any] | None`.
- **ActionStage** — drift-feedback exceptions now logged at WARNING with
  `exc_info=True`. `on_stop` passed at construction time only (private attribute).
- **errors.py** — added `ProfileNotFoundError(ModelLoadError)` for missing calibration
  profiles. `__reduce__` stringifies `cause` to survive pickling.

---

## [Unreleased]

### Fixed
- **Calibration "Faccia non rilevata" on Python 3.14 / MediaPipe 0.10.33** —
  `FacePresenceDetector` migrated from the legacy `mp.solutions.face_detection`
  API (not shipped on the Python 3.14 MediaPipe wheel) to the supported
  `mediapipe.tasks.vision.FaceDetector` (BlazeFace short-range, IMAGE mode).
  The BlazeFace model file (`blaze_face_short_range.tflite`, ~230 KB) is
  fetched on first calibration via the existing `ensure_model` utility and
  cached under `models/`.
- **Calibration "Faccia non rilevata"** — decoupled face presence from eyetrax's
  blink detector.  `wait_for_face` now uses a dedicated `FacePresenceDetector`
  (MediaPipe BlazeFace) so a spurious `blink=True` flag no longer blocks calibration.
- `open_camera_dshow` now validates frame brightness after warmup and recovers by
  forcing auto-exposure before raising `CameraError`, preventing silent black-frame
  failures on cameras with exposure locked by another app.
- `run_calibration` now hard-fails on `PatchError` instead of logging a warning
  and proceeding with broken upstream eyetrax code (which guaranteed face-detection
  failure on Windows via MSMF + black canvas).
- `wait_for_face` applies `FramePreprocessor` (CLAHE + sharpening) to every frame
  so MediaPipe FaceDetection receives contrast-enhanced input under variable lighting.

### Added
- `gaze/face_presence.py` — `FacePresenceDetector` (BlazeFace short-range) +
  `FacePresence` dataclass; context-manager lifecycle, exception-safe, blink-agnostic.
- `pipeline/calibration.py` — `_PreprocessingCapture` proxy: wraps `cv2.VideoCapture`
  so eyetrax's internal calibration loop also receives CLAHE-enhanced frames.
- `pipeline/calibration.py` — `_probe_camera()` helper for brightness and resolution
  probing (reusable in tests without camera hardware).
- Contextual calibration UI messages: "Ambiente troppo scuro", "Immagine sfocata",
  "Avvicinati alla webcam", with remaining-seconds countdown.
- `CameraSettings.auto_exposure` (`"auto"` | `"manual"`, default `"auto"`) and
  `CameraSettings.min_brightness_ok` (default `15.0`) settings.
- `gaze/compat/eyetrax.py` — `_PatchTarget` NamedTuple with `fallbacks` field:
  alternative module paths are tried before raising `PatchError`, enabling forward
  compatibility with eyetrax versions that reorganise calibration symbols.
- 20 new tests (`tests/gaze/test_face_presence.py`,
  `tests/pipeline/test_calibration.py`); total 268, coverage 80.82 %.

---

## [0.4.0] — 2026-04-13

### Added
- `errors.py` — unified `GazeControlError` hierarchy: `CameraError`, `CalibrationError`, `ModelLoadError`, `GazeBackendError`, `PipelineStageError`
- `gaze/backend.py` — `GazeBackend` Protocol + `GazePrediction` dataclass; implementations: `EyetraxBackend`, `L2CSBackend`, `EnsembleBackend`
- `pipeline/action_stage.py` — `ActionStage`: window-manager dispatch, drift feedback, `CLOSE_APP` → engine stop
- `pipeline/engine.py` — `PipelineEngine` unified runner (`error_policy="continue"|"halt"`)
- `pipeline/qt_adapter.py` — `QtPipelineThread` (QThread wrapper, emits `frame_processed`)
- `logging_config.py` — `configure_logging(LoggingSettings)` via `dictConfig`, `RotatingFileHandler`, optional JSON with `python-json-logger`
- `settings.py` — `LoggingSettings` (level, format, rotation_mb, backup_count) and `GestureLabelsSettings` nested models
- `gesture/classifier.py` — `@runtime_checkable GestureClassifier` Protocol satisfied by `MLPClassifier` and `RuleClassifier`
- Optional dependency group `logging = ["python-json-logger>=2.0"]`

### Changed
- Runner unified on `PipelineEngine`; `GazeControlPipeline` is now an alias
- `cli.py` no longer mutates `os.environ` at import time; `_suppress_third_party_logs()` called inside `main()` only
- `cli.py` uses `configure_logging(settings.logging)` instead of `logging.basicConfig`
- `GazeStage.start()` loads backend + profile and raises `ModelLoadError` on failure
- `WindowManagerError` and `PatchError` now descend from `GazeControlError`
- `FrameContext.target_window` typed as `int | None` (Win32 HWND); `landmarks` and `hand_result` typed via `TYPE_CHECKING` guard
- Coverage `fail_under` raised from 70 → 80
- Version bumped to 0.4.0

### Removed (BREAKING)
- `pipeline/orchestrator.py` — replaced by `PipelineEngine`
- `config.py` — deprecated legacy-constants shim removed
- `gaze/eyetrax_patches.py` — deprecated shim removed (`gaze/compat/eyetrax.py` kept)

### Migration 0.3 → 0.4
```python
# Before (config.py shim):
from gazecontrol.config import CAMERA_INDEX, GESTURE_LABELS

# After:
from gazecontrol.settings import get_settings
s = get_settings()
camera_index = s.camera.index
gesture_labels = s.gesture_labels.labels
```

---

## [0.3.0] — 2026-04-13

### Added
- `src/` package layout — proper PEP 517 structure
- `pydantic-settings`-based configuration (`settings.py`) with env-var override (`GAZECONTROL_*`)
- `paths.py` — `platformdirs`-aware path resolution; no more hardcoded `..` paths
- `pipeline/` package — orchestrator + stage decomposition (`CaptureStage`, `GazeStage`, `GestureStage`, `IntentStage`)
- `FrameContext` dataclass — typed container passed through pipeline stages
- `IntentState`, `GestureLabel`, `ActionType` enums — no more stringly-typed dispatch
- `@dataclass(frozen=True) Action` — typed action passed to `WindowManager`
- `FeatureSet` dataclass — replaces raw dict with dead `_thumb_tip_y` key
- `GazeMapper` persistence as `.npz` + meta JSON (removes fragile sklearn pickle)
- `GazeMapper` migration helper for old `.pkl` files
- `ModelLoadError` exception on explicit L2CS load failure
- L2CS ensemble wired: `crop_from_landmarks` now called with real MediaPipe landmarks
- `WindowManagerError` — replaces bare `except Exception: pass` blocks
- GitHub Actions CI (ubuntu + windows, Python 3.11 + 3.12)
- `pre-commit` hooks: ruff, mypy, end-of-file-fixer, detect-private-key
- `docs/architecture.md` — ASCII pipeline diagram
- `LICENSE` (MIT), `README.md`, `CONTRIBUTING.md`
- `tools/benchmark_pipeline.py` — stage latency baseline with regression CI gate
- `tools/train_gesture_mlp.py` — moved from runtime package
- 14+ new test files, coverage target >70%
- `pytest.mark.win32` — Win32 tests skip cleanly on Linux CI

### Fixed
- `test_state_machine.py` — broken `cfg` NameError on lines 39, 62, 131
- `FrameGrabber` thread race: `_capture_loop` read of `_cap` without lock
- `FrameGrabber.actual_resolution` calling `_cap.get()` under contention
- Double frame copy per loop tick (`read_bgr` + `read` → single `cvtColor`)
- `GazeMapper.predict` `AttributeError` when `_scaler` not yet assigned
- `GazeMapper` returns `None` (not a geometric estimate) when unfitted
- `L2CSModel` silent fail now sets `enabled=False` + `WARNING` log
- Pinch-release debounce: requires N≥3 consecutive non-PINCH frames
- All 6 bare `except Exception: pass` in `WindowsManager` replaced with typed handler
- `model_downloader` — SHA256 checksum, timeout, atomic write via `.part` temp file
- `HandDetector` — `detect_for_video(image, ts_ms)` with monotonic timestamp
- `logging.basicConfig` moved from import time into `cli.main()`
- `time.monotonic()` used consistently throughout pipeline (removed `time.time()` mix)
- Removed stale `BUG-*` / `ARCH-*` annotation comments from source
- `eyetrax_patches` moved to `gaze/compat/eyetrax.py`, hard-fails on untested version
- `mlp_classifier.train_and_export` removed from runtime package

### Changed
- `main.py` (611 LOC god-file) split into `cli.py` + `pipeline/`
- `config.py` replaced by `settings.py`; `config.py` retained as deprecated shim
- `WindowsManager` only implements primitives; base class owns full dispatch
- Import style normalized to absolute throughout (`from gazecontrol.x import Y`)
- All magic numbers promoted to named settings fields

---

## [0.2.0] — 2026-04 (pre-refactor baseline)

Initial enterprise-upgrade iteration: OneEuroFilter, I-VT FixationDetector,
DriftCorrector, L2CSModel ONNX wrapper, FramePreprocessor CLAHE, PipelineProfiler.

## [0.1.0] — 2026-02

Initial prototype: TinyMLP calibration, basic hand gesture rule classifier,
PyQt6 overlay, Win32 window management.
