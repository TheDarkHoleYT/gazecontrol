"""One-shot migrator for pre-v1.0 gaze calibration profiles.

Per ADR-0009 the v1.0+ on-disk layout is::

    <profiles>/<user_id>/<monitor_id>/v{N}.npz
    <profiles>/<user_id>/<monitor_id>/v{N}.meta.json
    <profiles>/<user_id>/<monitor_id>/latest.txt   (one line: "v{N}")

Versions before v1.0 wrote a flat file::

    <profiles>/<name>.gaze.npz
    <profiles>/<name>.gaze.meta.json   (optional)

This module copies any flat ``*.gaze.npz`` + sibling ``*.meta.json`` into
the new tree without deleting the original. The runtime keeps reading
the old path for backward-compat; users opt into the new layout by
running ``gazecontrol --migrate-profiles`` once.

Safety properties (matches ADR-0003 / atomic-write conventions):

- ``.part`` staging + ``os.replace`` rename (no half-written files).
- Idempotent: running twice is a no-op when the destination already
  exists with matching content.
- Read-only by default unless ``dry_run=False`` (the CLI defaults to
  ``False``; the dry-run flag flips it).
- Never modifies the source file.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import shutil
from pathlib import Path

from gazecontrol.paths import Paths

logger = logging.getLogger(__name__)

#: Filename suffix written by v0.x calibration. Matches
#: ``Paths.gaze_profile(name)`` output.
_LEGACY_SUFFIX = ".gaze.npz"

#: Default monitor bucket name used when migrating legacy flat profiles —
#: callers know nothing about which monitor the calibration belonged to,
#: so we tag them as "primary-legacy" until the user recalibrates with the
#: v1.0 multi-monitor flow.
DEFAULT_LEGACY_MONITOR = "primary-legacy"


@dataclasses.dataclass(frozen=True)
class MigrationResult:
    """One entry returned per discovered legacy profile.

    Attributes:
        source:        Path to the legacy ``*.gaze.npz`` that was discovered.
        target:        Path the migrator wrote (or would write) the v2 copy to.
        action:        One of ``"migrated"`` (file copied), ``"skipped"``
                       (destination already present and matched the source),
                       ``"dry_run"`` (no I/O performed), or ``"error"``
                       (copy failed — see ``message``).
        message:       Human-readable detail. Empty for ``"migrated"`` and
                       ``"dry_run"``; populated for ``"skipped"`` and
                       ``"error"``.
    """

    source: Path
    target: Path
    action: str
    message: str = ""


def migrate_profiles(
    profiles_dir: Path | None = None,
    *,
    user_id: str = "default",
    monitor_id: str = DEFAULT_LEGACY_MONITOR,
    dry_run: bool = False,
) -> list[MigrationResult]:
    """Migrate every flat ``*.gaze.npz`` under *profiles_dir* into the v2 tree.

    Args:
        profiles_dir: Override for the profiles root (defaults to
                      :func:`Paths.profiles`). Useful for tests.
        user_id:      User bucket to migrate into (default ``"default"``).
        monitor_id:   Monitor bucket to migrate into (default
                      ``"primary-legacy"`` — see ADR-0009 §migration).
        dry_run:      When True, report what *would* be migrated without
                      touching the filesystem. Useful for ``--migrate-profiles
                      --dry-run`` from the CLI.

    Returns:
        One :class:`MigrationResult` per discovered legacy profile, in
        deterministic (sorted) order. Empty list when no legacy files
        exist (already on v2 or fresh install).
    """
    root = Path(profiles_dir) if profiles_dir is not None else Paths.profiles()
    if not root.is_dir():
        logger.info("profile_migrate: profiles dir %s does not exist; nothing to do.", root)
        return []

    legacy = sorted(p for p in root.glob(f"*{_LEGACY_SUFFIX}") if p.is_file())
    if not legacy:
        logger.info("profile_migrate: no legacy %s files under %s.", _LEGACY_SUFFIX, root)
        return []

    results: list[MigrationResult] = []
    for src in legacy:
        # Strip the ".gaze.npz" suffix to recover the profile name.
        name = src.name[: -len(_LEGACY_SUFFIX)] or "default"
        dst_dir = root / user_id / monitor_id
        dst_npz = dst_dir / "v1.npz"
        dst_meta = dst_dir / "v1.meta.json"
        dst_latest = dst_dir / "latest.txt"
        src_meta = src.with_suffix("").with_suffix(".meta.json")
        # ``Path.with_suffix`` strips only the last suffix; ``foo.gaze.npz``
        # → ``foo.gaze`` → ``foo.meta.json``. The original layout uses
        # ``foo.gaze.meta.json``, so look for both.
        legacy_meta_candidates = [
            src.parent / f"{name}.gaze.meta.json",
            src.parent / f"{name}.meta.json",
            src_meta,
        ]
        src_meta_actual: Path | None = next(
            (p for p in legacy_meta_candidates if p.exists()), None
        )

        if dst_npz.exists() and _same_bytes(src, dst_npz):
            results.append(
                MigrationResult(
                    source=src,
                    target=dst_npz,
                    action="skipped",
                    message=f"destination already up-to-date for profile {name!r}",
                )
            )
            continue

        if dry_run:
            results.append(
                MigrationResult(
                    source=src,
                    target=dst_npz,
                    action="dry_run",
                    message=f"would migrate profile {name!r} → {dst_npz}",
                )
            )
            continue

        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            _atomic_copy(src, dst_npz)
            if src_meta_actual is not None:
                _atomic_copy(src_meta_actual, dst_meta)
            _atomic_write_text(dst_latest, "v1\n")
        except OSError as exc:
            logger.exception("profile_migrate: copy failed for %s", src)
            results.append(
                MigrationResult(
                    source=src,
                    target=dst_npz,
                    action="error",
                    message=str(exc),
                )
            )
            continue

        logger.info("profile_migrate: %s → %s", src, dst_npz)
        results.append(
            MigrationResult(source=src, target=dst_npz, action="migrated", message="")
        )

    return results


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy *src* to *dst* via a ``.part`` staging file + ``os.replace``."""
    part = dst.with_suffix(dst.suffix + ".part")
    try:
        shutil.copyfile(src, part)
        os.replace(part, dst)
    finally:
        if part.exists():
            with contextlib.suppress(OSError):
                part.unlink()


def _atomic_write_text(dst: Path, content: str) -> None:
    """Write *content* to *dst* via ``.part`` + ``os.replace``."""
    part = dst.with_suffix(dst.suffix + ".part")
    try:
        part.write_text(content, encoding="utf-8")
        os.replace(part, dst)
    finally:
        if part.exists():
            with contextlib.suppress(OSError):
                part.unlink()


def _same_bytes(a: Path, b: Path) -> bool:
    """Return True when *a* and *b* are byte-for-byte identical.

    Used for idempotency: if the destination already matches the source
    we skip rather than overwrite. Reads in 1 MB chunks so it works on
    large ONNX-derived profiles without loading both files into RAM.
    """
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
    except OSError:
        return False
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            chunk_a = fa.read(1 << 20)
            chunk_b = fb.read(1 << 20)
            if chunk_a != chunk_b:
                return False
            if not chunk_a:
                return True
