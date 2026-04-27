# ADR-0005 — Process-wide crash handler stack

Status: Accepted (v0.8.0).

## Context

C/C++ crashes in MediaPipe or ONNX, uncaught background-thread
exceptions, and SIGTERM during a soak run all left zero forensic
artefacts in v0.7.x.

## Decision

`gazecontrol.runtime.crash.install_crash_handlers()` registers, in order:

1. `sys.excepthook` — main-thread fall-through → `CRITICAL` log.
2. `threading.excepthook` — background-thread fall-through → `CRITICAL`.
3. `faulthandler.enable(file=fault.log, all_threads=True)` — C-level
   tracebacks for segfaults / native crashes.
4. POSIX `SIGTERM` (and Windows `SIGBREAK`) → graceful shutdown
   callback (`engine.request_stop`).
5. `qInstallMessageHandler` (when PyQt6 is loaded) — Qt warnings flow
   into the same logger.

Installed twice during `cli.main`: once before the engine is built (for
import-time crashes) and once after (to register the engine as the
shutdown callback).

## Consequences

- Every crash leaves a log line; ops can root-cause from `gazecontrol.log`.
- Idempotent — safe to call from `cli.main` and from tests via
  `uninstall_crash_handlers()` to restore stdlib defaults.
