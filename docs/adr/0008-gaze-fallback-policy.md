# ADR-0008 — Gaze backend failure → hand-only fallback policy

Status: Accepted (v1.0.0).

## Context

Until v0.8.x a mid-session failure in the gaze stage (ONNX exception, face
detector returning `None` for many frames, BlazeFace model unloaded, etc.)
left the pipeline running with `gaze_xy=None`. `PointerFusionStage` already
falls back to the hand pointer when gaze is missing, but there were two
problems:

1. No bound on how long the pipeline tolerates a degraded gaze backend —
   `None` could continue forever, silently masking a broken installation.
2. No way for an enterprise deployment to choose between "keep limping"
   and "stop loudly" — useful when gaze is mandatory (accessibility) vs.
   optional (productivity).

## Decision

Introduce `FusionSettings.gaze_failure_policy: Literal["continue", "hand_only", "stop"] = "hand_only"`.

`pipeline/gaze_stage.py` tracks consecutive backend failures
(exception or `None` return) in a private counter. When the counter exceeds
`gaze_failure_threshold_frames` (default `10`, ~330 ms at 30 fps) the stage
takes one of three actions:

| Policy       | Action |
|--------------|--------|
| `continue`   | Log INFO once; keep emitting `gaze_xy=None`. Legacy behaviour. |
| `hand_only`  | Set `FrameContext.gaze_backend_down = True`; emit profiler counter `gazecontrol_backend_fallback_total{backend}`; HUD shows degraded badge. `PointerFusionStage` already routes to hand-only when gaze is None. |
| `stop`       | Raise `GazeBackendError` and request engine shutdown via the registered crash-handler callback (exit code 12). |

A successful prediction resets the counter to zero. A recovery transition
(`gaze_backend_down=True` → first successful prediction) emits a
`gaze.recovered` structured log event.

The `gaze_failure_threshold_frames` and `gaze_failure_policy` settings are
exposed in `GazeSettings` of `runtime.toml`, so deployments can tune the
threshold without code changes.

## Consequences

- No silent degradation: users see a HUD badge within ~330 ms of gaze loss.
- Operators choose the policy that matches their risk model.
- Hand-only is the new default — the most common case (gaze is auxiliary)
  is also the least surprising.
- A regression test (`tests/pipeline/test_gaze_fallback.py`) simulates
  10 consecutive backend exceptions and asserts the context flag flips and
  the profiler counter increments by 1.
- `gazecontrol_backend_fallback_total` joins the existing Prometheus surface
  documented in the profiler docs.
