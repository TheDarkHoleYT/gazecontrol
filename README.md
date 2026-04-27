# GazeControl

Desktop control via **hand gestures**. Pinch to click, hold to drag,
two fingers to scroll — all from a webcam.

> Eye tracking (gaze-assisted pointer) is on the roadmap. The code
> path exists behind an optional extra and a startup mode selector,
> but it requires per-user calibration and is not production-ready.

---

## Quick start

```bash
pip install gazecontrol
gazecontrol
```

That's it — webcam in, cursor out. Default mode is **hand-only**;
no calibration required.

---

## Hardware requirements

| Item     | Minimum             |
|----------|---------------------|
| Webcam   | 720p @ 30 fps       |
| OS       | Windows 10 / 11     |
| GPU      | None required       |

Win32-only for now (window manager uses native Windows APIs).

---

## Installation

```bash
# Production
pip install gazecontrol

# Development
git clone https://github.com/<you>/gazecontrol
cd gazecontrol
pip install -e ".[dev]"
pre-commit install
```

Run `gazecontrol --doctor` to verify the camera and dependencies.

---

## Gestures

| Gesture                   | Action                          |
|---------------------------|---------------------------------|
| Index pinch (tap)         | Left click                      |
| Index pinch (held)        | Drag the hovered window         |
| Index pinch in corner     | Resize the hovered window       |
| Two fingers up / down     | Scroll                          |
| Double pinch              | Toggle the app launcher         |

---

## Configuration

Settings load from environment variables (prefix `GAZECONTROL_`,
double underscore for nested groups) and an optional `settings.toml`
in the working directory. See `settings.toml.example` for the full
surface.

```bash
# Example: bump pinch threshold
export GAZECONTROL_INTERACTION__PINCH_DOWN_THRESHOLD=0.04
```

---

## CLI

```text
gazecontrol                       # run hand-only pipeline
gazecontrol --no-overlay          # headless run (no Qt HUD)
gazecontrol --doctor              # probe camera + deps
gazecontrol --dump-config         # dump effective settings as JSON
gazecontrol --benchmark 30        # run 30 s, print per-stage latency
```

---

## Roadmap — eye tracking (experimental)

An `eye-hand` input mode exists behind the `[eye]` extra. It pairs
the hand pipeline with an L2CS-Net + eyetrax gaze ensemble and a
PointerFusionStage that lets gaze drive target selection while hand
keeps click/drag precision.

Status: works on the maintainer's machine but needs per-user
calibration, an unbundled L2CS ONNX model, and tighter drift
correction. Not recommended for end users yet. Track progress in
[docs/architecture.md](docs/architecture.md).

---

## Project layout

```
src/gazecontrol/
├── cli.py                # entry point + doctor + benchmark
├── runtime/              # input mode + pipeline factory
├── pipeline/             # CaptureStage → GestureStage → ...
├── gesture/              # MediaPipe + rule/MLP classifiers
├── interaction/          # InteractionFSM + WindowHitTester
├── filters/              # 1€, Kalman, dead-zone, accel curve
├── overlay/              # PyQt6 HUD
├── window_manager/       # Win32 wrappers
├── gaze/                 # roadmap: eye-tracking backends
└── settings.py           # pydantic-settings
```

See [docs/architecture.md](docs/architecture.md) for the full
diagram and ADRs.

---

## License

MIT.
