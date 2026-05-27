# ADR-0007 — L2CS-Net ONNX supply-chain pinning

Status: Accepted (v1.0.0).

## Context

L2CS-Net is the appearance-based gaze backend behind `EnsembleBackend`.
Until v0.8.x users obtained the weights by running `tools/download_l2cs.py`,
which fetched a non-deterministic PyTorch `.pkl` from the upstream research
repo and converted it to ONNX on the user's machine. Three problems:

1. The `.pkl` file is not pinned — upstream can rewrite history.
2. The converter is unsandboxed Python; a malicious `.pkl` runs arbitrary
   code via `torch.load`.
3. ONNX layer ordering depends on the local PyTorch version, so two users
   converting the same `.pkl` get binaries with different SHA256s. CI can't
   detect tampering.

The other three model assets (`face_landmarker.task`, `hand_landmarker.task`,
`blaze_face_short_range.tflite`) already follow ADR-0003 pinning.

## Decision

Add `l2cs_net_gaze360.onnx` to `utils/model_downloader.py::_MODELS` with a
pinned SHA256. The canonical FP16 ONNX is published as a release asset on
the GazeControl GitHub release page; the URL is HTTPS-only and tracked in
the registry alongside the other three models.

`L2CSBackend.start()` calls `ensure_model("l2cs_net_gaze360.onnx", Paths.models())`
instead of the existence-check path that exists today. Failure to download or
mismatch on SHA256 raises `ModelDownloadError`, which the pipeline maps to
`gaze_failure_policy` (ADR-0008).

`tools/download_l2cs.py` is retained behind a `--from-source` flag for
researchers who need to retrain. It writes to a sibling path
(`l2cs_net_gaze360.from_source.onnx`) and never overwrites the pinned file.

Rotation policy: when a new ONNX is published, bump the SHA256 in
`_MODELS` *and* the model filename suffix (e.g. `l2cs_net_gaze360_v2.onnx`).
The downloader's atomic write + verify path then handles cache eviction.

## Consequences

- Reproducible deployments: every install of v1.0+ runs bit-identical L2CS
  weights.
- The escape hatch `GAZECONTROL_ALLOW_UNPINNED_MODELS=1` from ADR-0003
  remains the only path to run an unpinned binary, with the same WARNING.
- Adding a new gaze model requires committing the digest in code review —
  CI guards via `tests/utils/test_model_downloader.py`.
- Users of `tools/download_l2cs.py --from-source` get a clearly-separate
  file path so they cannot accidentally shadow the pinned blob.
