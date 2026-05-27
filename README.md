# GazeControl

Desktop control via **hand gestures** + **eye tracking**. Pinch to
click, hold to drag, two fingers to scroll — and (in `eye-hand` mode)
let your gaze pick the target while the hand keeps click precision.

All on-device, all on a webcam. See [`PRIVACY.md`](PRIVACY.md) for the
data-handling contract; the v1.0 enterprise upgrade closed 24 plan
gaps and is documented in [`CHANGELOG.md`](CHANGELOG.md).

---

## Quick start

```bash
pip install gazecontrol
gazecontrol
```

That's it — webcam in, cursor out. Default mode is **hand-only**;
no calibration required.

For eye tracking, install the `[eye]` extra, run `--calibrate-gaze`
once, then `--mode eye-hand`:

```bash
pip install "gazecontrol[eye]"
gazecontrol --calibrate-gaze       # 13-point grid, ~30 s
gazecontrol --mode eye-hand
gazecontrol --calibrate-incremental 3   # later top-up after a head shift
```

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
# Run modes
gazecontrol                        # default; hand-only unless persisted otherwise
gazecontrol --mode hand            # force hand-only
gazecontrol --mode eye-hand        # force eye + hand fusion
gazecontrol --no-overlay           # headless run (no Qt HUD)

# Diagnostics
gazecontrol --doctor               # probe camera + deps (existence only)
gazecontrol --doctor --functional  # also run live inference on dummy frames (G13)
gazecontrol --healthcheck          # one-shot probe with stable exit codes
gazecontrol --dump-config          # dump effective settings as JSON

# Performance + replay
gazecontrol --benchmark 30                          # 30 s headless, print percentiles
gazecontrol --benchmark 10 --bench-json bench.json  # also emit JSON (CI hook, G11)
gazecontrol --benchmark 10 --bench-mock             # synthetic camera, no webcam needed

# Calibration (eye-hand mode)
gazecontrol --calibrate-gaze                        # full 13-point grid + holdout
gazecontrol --calibrate-incremental 3               # 3 / 5 / 9 / 13-point top-up
gazecontrol --migrate-profiles                      # one-shot v1 → v2 profile layout

# Compliance
gazecontrol --purge-profiles [--yes]                # GDPR Art.17 erasure (G16)
```

Per-module log levels: `--log-modules gazecontrol.gaze:DEBUG`.
Localised UI strings: `GAZECONTROL_LOCALE=en` (also `LANG` aware).

---

## Eye tracking — v1.0 GA

`eye-hand` mode pairs the hand pipeline with an L2CS-Net + eyetrax
gaze ensemble and a `PointerFusionStage` that lets gaze drive target
selection while the hand keeps click / drag precision. The v1.0
release closes 24 enterprise gaps tracked in
[`CHANGELOG.md`](CHANGELOG.md):

- **Accuracy**: per-frame confidence model, head-pose PnP into the
  mapper, Gaussian-process mapper that exposes `uncertainty_px`,
  Kalman ensemble fusion, multi-face tracking, EAR blink detection,
  Kalman drift corrector + explicit recenter.
- **Calibration UX**: 13-point grid + holdout error reporting,
  incremental top-up (`--calibrate-incremental N`), per-user +
  per-monitor profile tree
  ([ADR-0009](docs/adr/0009-multi-monitor-profile-schema.md)).
- **Ops**: pinned L2CS ONNX in the
  [supply-chain registry](docs/adr/0007-l2cs-onnx-pinned.md), CI
  latency SLA gate, replay regression harness, per-frame structured
  telemetry, configurable gaze→hand fallback policy
  ([ADR-0008](docs/adr/0008-gaze-fallback-policy.md)).
- **Compliance**: [`PRIVACY.md`](PRIVACY.md), GDPR Art.17
  `--purge-profiles` command, per-app fusion threshold overrides,
  minimal i18n (en / it).

The first-time L2CS model bootstrap still goes through
`tools/download_l2cs.py` (Google Drive → ONNX conversion) until the
canonical signed release on the GazeControl GitHub release page
lands; thereafter the model_downloader picks it up automatically per
[ADR-0007](docs/adr/0007-l2cs-onnx-pinned.md).

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
├── gaze/                 # eye-tracking backends + pure helpers
│   │                     # (confidence, blink, head_pose, face_tracking,
│   │                     # face_cascade, monitor_id, drift_corrector)
└── settings.py           # pydantic-settings
```

See [docs/architecture.md](docs/architecture.md) for the full
diagram and ADRs.

---

## License

GazeControl is released under the **MIT License** — see
[`LICENSE`](LICENSE).

Bundled or downloaded components from third-party projects (L2CS-Net,
MediaPipe Tasks) retain their upstream licenses — see
[`NOTICE.md`](NOTICE.md) for attributions and full text.
