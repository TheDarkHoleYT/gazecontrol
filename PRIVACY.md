# Privacy & Data Handling

GazeControl is an **on-device** accessibility tool. The runtime never
sends camera frames, gaze samples, calibration data, or telemetry off
the machine. This document describes exactly what data is processed,
where it lives on disk, and how users can purge it.

This file is the canonical reference for the v1.0 enterprise privacy
posture. The matching design decisions are recorded in
[ADR-0003](docs/adr/0003-model-sha256-pinning.md) (supply-chain) and
[ADR-0007](docs/adr/0007-l2cs-onnx-pinned.md) (model pinning).

---

## 1. Data processed in memory

| Source            | What                                              | Where it lives                                |
|-------------------|---------------------------------------------------|-----------------------------------------------|
| Webcam            | Live frames (BGR + RGB)                           | Pipeline RAM — discarded each tick.           |
| Hand landmarker   | 21 hand-keypoint coords + features                | Pipeline RAM — discarded each tick.           |
| Face landmarker   | 468 face-mesh keypoints (when EYE_HAND mode)      | Pipeline RAM — discarded each tick.           |
| Face detector     | BlazeFace bounding box (when EYE_HAND mode)       | Pipeline RAM — discarded each tick.           |
| Gaze backends     | (yaw, pitch) angles, head pose, EAR               | Pipeline RAM — discarded each tick.           |

**Frames are never written to disk.** The pipeline keeps only the most
recent frame in a single buffer and overwrites it on each tick.

---

## 2. Data persisted to disk

GazeControl writes four categories of files. All paths resolve via
``platformdirs`` so they obey OS conventions (XDG on Linux, AppData on
Windows, Application Support on macOS).

| Path                                            | Purpose                                                                      |
|-------------------------------------------------|------------------------------------------------------------------------------|
| ``<user_config>/gazecontrol/profiles/``         | Gaze calibration profiles (.gaze.npz + .meta.json + v2 tree).               |
| ``<user_config>/gazecontrol/launcher.toml``     | User-edited application launcher list.                                       |
| ``<user_config>/gazecontrol/runtime.toml``      | Persisted input-mode choice (HAND_ONLY vs EYE_HAND).                         |
| ``<user_log>/gazecontrol/gazecontrol.log``      | Rotating log file; one fault dump (``faulthandler.log``) alongside it.       |

### 2.1 What the profile contains

Calibration profiles (``v{N}.npz`` + ``v{N}.meta.json``) hold:

- **Mapper coefficients** — numerical weights that map (yaw, pitch) →
  screen pixels.
- **Inline training samples** — the (yaw, pitch) angles and the
  screen-pixel targets captured during calibration (typically ~200
  floats, ~3 KB). Required so ``--calibrate-incremental`` can refit
  without re-running the full grid.
- **Metadata** — ISO-8601 calibration timestamp, sample count,
  user_id, monitor_id, fit method.

**Profiles do NOT contain:**

- Any camera frames or raw eye crops.
- Any identifying biometric template (no face descriptors, no
  embeddings — only abstract angle → pixel coefficients).
- Any network identifier or system metadata beyond the user-chosen
  ``user_id`` / ``monitor_id`` strings.

A profile is identifying only insofar as it is associated with a
user-chosen ``user_id``; the default is the literal string ``default``.

### 2.2 What the logs contain

The default log handler is ``RotatingFileHandler`` capped at
``5 MB × 5 backups = 25 MB max`` (see ``LoggingSettings.rotation_mb``
and ``backup_count`` in ``src/gazecontrol/settings.py``). Lines record:

- Stage timing percentiles (no frame content).
- Counters: frame totals, dropped frames, backend fallbacks.
- Confidence values, fixation events, drift offsets.
- One-line incident messages (camera reopen, model load failures).

The opt-in JSON ``gaze.pred`` per-frame record (G15) additionally logs
gaze pixel coordinates, head pose angles, and quality flags. It is
**off by default** and never enabled in production without explicit
``LoggingSettings.telemetry_per_frame = True``.

Logs **do NOT contain** raw frames, calibration training samples, or
user names beyond the chosen ``user_id``.

---

## 3. Network policy

GazeControl makes outbound HTTPS connections in exactly two cases:

1. **On first run**, the model downloader fetches three pinned files
   from ``storage.googleapis.com`` (MediaPipe) and one from the
   GazeControl GitHub release page (L2CS-Net, pending v1.0 release).
   Every file is SHA256-verified before use; mismatched downloads
   are deleted and the call fails closed.
2. **CodeQL / pip-audit / CycloneDX SBOM** during CI (developer
   machines only, never at runtime).

After the initial model download (or when the operator pre-populates
``models/`` from an air-gapped mirror), GazeControl operates entirely
offline.

There is **no analytics, no crash reporter, no telemetry endpoint, no
auto-update**.

---

## 4. GDPR Art. 17 — right to erasure

Users may erase all locally-stored GazeControl data with a single
command:

```bash
gazecontrol --purge-profiles
```

The command prompts for confirmation unless ``--yes`` is passed, then
recursively deletes ``<user_config>/gazecontrol/profiles/``, removes
``runtime.toml``, and emits a ``compliance.purge`` log record so an
audit trail (the log file itself, also user-owned) records the action.

Logs older than the rotation policy are deleted automatically; the
operator may also remove ``<user_log>/gazecontrol/`` manually.

---

## 5. Model provenance

| Model                                | Source                                | Pinned via   |
|--------------------------------------|---------------------------------------|--------------|
| ``face_landmarker.task``             | Google MediaPipe (Apache 2.0)         | ADR-0003     |
| ``hand_landmarker.task``             | Google MediaPipe (Apache 2.0)         | ADR-0003     |
| ``blaze_face_short_range.tflite``    | Google MediaPipe (Apache 2.0)         | ADR-0003     |
| ``l2cs_net_gaze360.onnx``            | L2CS-Net (Abdelrahman & Hossny, 2022) | ADR-0007     |

The L2CS-Net model is downloaded from the GazeControl GitHub release
page once a v1.0 signed release exists. Until then users bootstrap via
``tools/download_l2cs.py`` from the upstream Google Drive folder
documented in the L2CS-Net paper.

---

## 6. Reporting a vulnerability

Open a GitHub issue with the ``security`` label or email
``hertachecchannel@gmail.com``. Do not include exfiltrated user data
in the report.
