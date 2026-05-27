# ADR-0009 — Multi-monitor + multi-user gaze profile schema

Status: Accepted (v1.0.0).

## Context

v0.7/0.8 stored a single gaze calibration profile under a flat name
(`default.gaze.npz` + `default.meta.json`), with `screen_w`/`screen_h`
recorded but no notion of *which* monitor or *which* user the calibration
belongs to. Two pain points:

1. Multi-monitor setups (e.g. 4K laptop + 1080p external) calibrated on
   the primary screen drift badly on the secondary — the mapper assumes
   the same `screen_w`/`screen_h` everywhere.
2. Shared workstations and lab environments cannot maintain per-user
   calibrations without manually managing files outside the app.

Additionally, the v1 `.npz` does not persist the *training* (angles,
targets) used to fit the mapper, which blocks incremental recalibration
(`partial_fit`).

## Decision

Profile layout becomes a directory tree under `Paths.profiles()`:

```
profiles/
└── <user_id>/
    └── <monitor_id>/
        ├── v1.npz       (older versions, retained until purged)
        ├── v2.npz
        ├── v2.meta.json
        └── latest.txt   (single line: "v2")
```

- `user_id` defaults to `"default"`. CLI flag `--user <id>` overrides.
- `monitor_id` is the SHA1-truncated hash of
  `QGuiApplication.screens()[i].name()` (stable across reboots on Windows).
- `latest.txt` is a Windows-safe alternative to symlinks; readers parse
  one line, callers write atomically (`.part` rename).

`GazeMapper._FORMAT_VERSION` becomes `"2"`. The `.npz` adds:

| Key                | Purpose                                          |
|--------------------|--------------------------------------------------|
| `training_angles`  | (N, 2) calibration yaw/pitch samples            |
| `training_targets` | (N, 2) calibration screen-pixel targets         |
| `training_head_poses` | (N, 3) optional head pose at calibration time |
| `kernel_dual_coef` | KernelRidge / GP dual coefficients (optional)    |
| `kernel_X_fit`     | KernelRidge / GP support points (optional)       |

The `meta.json` adds:

| Field             | Purpose                                       |
|-------------------|-----------------------------------------------|
| `schema_version`  | `2` (load-time auto-default = `"1"`)          |
| `calibrated_at`   | ISO-8601 UTC                                  |
| `samples_count`   | N samples used for fit                        |
| `loo_error_px`    | LOO-CV error in pixels                        |
| `holdout_error_px`| Held-out validation error in pixels           |
| `monitor_id`      | Hash matching directory name                  |
| `user_id`         | Matching directory name                       |
| `fit_method`      | `"9pt" \| "13pt_holdout" \| "incremental_3pt" \| "incremental_5pt"` |
| `mapper_type`     | `"poly_ridge" \| "kernel_ridge" \| "gp"`     |
| `feature_schema`  | List of feature names used (for predict-side reconstruction) |

`GazeStage` watches the active monitor under cursor and swaps the loaded
mapper on monitor change with a 250 ms HUD blink to suppress flicker.
Drift offsets become per-monitor (a dict in `DriftCorrector`).

A one-shot CLI migrator `gazecontrol profile migrate` walks any pre-v1.0
flat `*.gaze.npz` files under `Paths.profiles()` and rewrites them into
the new tree as `default/<primary_monitor_id>/v1.npz`, preserving the
original until the user removes it. Loading a v1 schema works in-place:
`GazeMapper.load()` fills the new metadata with defaults and logs a
one-time INFO suggesting recalibration.

## Consequences

- Breaking layout change → major bump (v1.0.0). One-shot migrator covers
  the only non-trivial upgrade path.
- Multi-monitor + multi-user setups are first-class.
- Incremental recalibration (`partial_fit`) becomes possible because the
  training data is now persisted inline.
- Storage footprint grows modestly: ~3 KB per profile (200 samples × 16 B
  × 2 arrays).
- Tests added under `tests/gaze/test_profile_schema_v2.py`:
  - v1 file loads with defaults filled.
  - v2 file round-trips through save/load with all new fields.
  - `profile migrate` is idempotent.
