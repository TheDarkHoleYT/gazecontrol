# Changelog

All notable changes to GazeControl are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
