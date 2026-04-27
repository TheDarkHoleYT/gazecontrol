# ADR-0003 — Model SHA256 pinning policy

Status: Accepted (v0.7.1).

## Context

The runtime depends on three downloaded models (MediaPipe HandLandmarker,
FaceLandmarker, BlazeFace). A swap of these binaries can silently change
behaviour or open a supply-chain hole.

## Decision

`utils/model_downloader.py` keeps a pinned registry of `(name, url, sha256)`
tuples and refuses to use any blob whose digest does not match. The
escape hatch `GAZECONTROL_ALLOW_UNPINNED_MODELS=1` is debug-only and
emits a WARNING.

## Consequences

- Downloads are deterministic and verifiable.
- Adding a new model requires committing the digest in code review.
- CI enforces this implicitly via the test suite + SBOM job.
