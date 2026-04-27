# ADR-0004 — mypy strict, no per-module overrides

Status: Accepted (v0.8.0).

## Context

In v0.7.x `[tool.mypy.overrides]` opted half the codebase out of strict
mode. The result was 312 hidden errors and an illusory typing posture.

## Decision

Top-level `tool.mypy` declares `strict = true` plus `warn_unreachable`,
`warn_redundant_casts`, `disallow_untyped_defs`, and
`disallow_any_generics`. The overrides block is empty.

`# type: ignore` comments must carry the specific error code
(`# type: ignore[attr-defined]`); bare ignores are caught by
`warn_unused_ignores`.

`src/gazecontrol/py.typed` is shipped (PEP 561).

## Consequences

- New code must be fully typed; CI rejects untyped defs.
- Untyped third-party deps (mediapipe, onnxruntime, eyetrax) are
  isolated behind `Any`-typed adapters at the boundary, never leaking
  inward.
- Downstream packages get accurate type information out of the box.
