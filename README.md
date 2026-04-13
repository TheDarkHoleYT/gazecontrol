# GazeControl

Desktop control via eye tracking + hand gesture recognition on Windows.

## Features

- **Gaze tracking** — TinyMLP calibrated mapper + 1€ adaptive filter + I-VT fixation detection
- **L2CS-Net ensemble** — optional CNN-based gaze for higher accuracy (requires model download)
- **Gesture recognition** — MediaPipe hand landmarks → MLP + rule classifier (pinch, swipe, resize, ...)
- **Intent FSM** — finite-state machine translates gaze + gesture into window actions (move, resize, close, scroll)
- **Overlay HUD** — transparent PyQt6 overlay showing fixation state and debug info

## Requirements

- Windows 10/11 (64-bit)
- Python ≥ 3.11
- Webcam (1280×720 preferred)

## Installation

```bash
git clone <repo-url> gazecontrol
cd gazecontrol
python -m venv .venv && .venv\Scripts\activate
pip install -e .
```

For development (linting, tests, pre-commit):

```bash
pip install -e ".[dev]"
pre-commit install
```

## Quick start

**Step 1 — Calibrate** (25-point grid, ~60 s):

```bash
gazecontrol --calibrate --profile default
```

**Step 2 — Run**:

```bash
gazecontrol --profile default
```

**Optional: download L2CS-Net** for higher accuracy:

```bash
python tools/download_l2cs.py
```

## Configuration

Copy `settings.toml.example` to `settings.toml` and adjust:

```toml
[camera]
index = 0
width = 1280
height = 720

[gaze]
model_mode = "mlp"  # "mlp" | "l2cs" | "ensemble"
```

Override any setting via env var with prefix `GAZECONTROL_`:

```bash
GAZECONTROL_CAMERA__INDEX=1 gazecontrol --profile default
```

## Benchmark

```bash
python tools/benchmark_gaze.py --profile default
python tools/benchmark_pipeline.py --baseline baselines/pipeline.json
```

## Project structure

```
src/gazecontrol/
├── cli.py                # CLI entry point
├── settings.py           # pydantic-settings configuration
├── paths.py              # platformdirs-based path resolution
├── pipeline/             # Stage-based pipeline architecture
│   ├── orchestrator.py   # GazeControlPipeline (QThread orchestrator)
│   ├── capture_stage.py  # Camera capture + preprocessing
│   ├── gaze_stage.py     # Gaze estimation + filtering + ensemble
│   ├── gesture_stage.py  # Hand detection + classification
│   ├── intent_stage.py   # FSM + window selection
│   └── calibration.py   # Calibration helpers
├── capture/              # FrameGrabber, FramePreprocessor
├── gaze/                 # OneEuroFilter, FixationDetector, DriftCorrector, L2CSModel, GazeMapper
├── gesture/              # HandDetector, FeatureExtractor, RuleClassifier, MLPClassifier
├── intent/               # IntentStateMachine, WindowSelector
├── window_manager/       # BaseWindowManager, WindowsManager (Win32)
├── overlay/              # OverlayWindow (PyQt6), HUDRenderer
└── utils/                # PipelineProfiler, ModelDownloader
tools/
├── download_l2cs.py      # Download + convert L2CS-Net ONNX
├── benchmark_gaze.py     # MAE / angular-error benchmark
├── benchmark_pipeline.py # Stage latency benchmark
└── train_gesture_mlp.py  # Train and export gesture MLP
tests/                    # pytest, >70% coverage
docs/                     # Design docs, architecture
```

See `docs/architecture.md` for the full pipeline diagram.

## License

MIT — see [LICENSE](LICENSE).
