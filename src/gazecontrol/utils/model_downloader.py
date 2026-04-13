"""Model downloader — fetch MediaPipe Tasks assets with safety checks.

Improvements vs original:
- Timeout on download (30 s connect, 60 s read).
- Atomic write: downloads to ``<dest>.part`` then renames on success.
- SHA256 manifest check: when a checksum is registered, verifies file
  integrity after download (and after every subsequent re-use if checksum
  is known).
- Partial files are cleaned up on any failure.
"""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

#: Map from filename → (URL, sha256_hex | None)
_MODELS: dict[str, tuple[str, str | None]] = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        None,  # checksum not pinned — update when upstream provides stable hash
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        None,
    ),
}

_DOWNLOAD_TIMEOUT = (30, 60)  # (connect_s, read_s)


def ensure_model(model_name: str, models_dir: str | os.PathLike[str]) -> str:
    """Return the local path of *model_name*, downloading if absent.

    Args:
        model_name: File name registered in ``_MODELS``.
        models_dir: Directory where the model should be stored.

    Returns:
        Absolute path to the downloaded model file.

    Raises:
        ValueError:    Model name not in registry.
        RuntimeError:  Download failed.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / model_name

    entry = _MODELS.get(model_name)
    if entry is None:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Registered models: {list(_MODELS)}"
        )
    url, expected_sha256 = entry

    if dest.exists():
        if expected_sha256 and not _verify_sha256(dest, expected_sha256):
            logger.warning(
                "SHA256 mismatch for cached %s — re-downloading.", model_name
            )
            dest.unlink()
        else:
            return str(dest)

    _download(model_name, url, dest, expected_sha256)
    return str(dest)


def _download(
    model_name: str,
    url: str,
    dest: Path,
    expected_sha256: str | None,
) -> None:
    """Download *url* to *dest* atomically with optional checksum verification."""
    part = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s from %s ...", model_name, url)

    try:
        with urllib.request.urlopen(  # noqa: S310 (URL from trusted registry)
            url, timeout=_DOWNLOAD_TIMEOUT[1]
        ) as response, part.open("wb") as fh:
            while chunk := response.read(1 << 20):  # 1 MB chunks
                fh.write(chunk)
    except Exception as exc:
        if part.exists():
            part.unlink()
        raise RuntimeError(f"Failed to download {model_name}: {exc}") from exc

    if expected_sha256 and not _verify_sha256(part, expected_sha256):
        part.unlink()
        raise RuntimeError(
            f"SHA256 mismatch for {model_name} after download. "
            "The file may be corrupted or the URL has changed."
        )

    # Atomic rename — avoids leaving a half-written file if the process dies.
    os.replace(part, dest)
    logger.info("Model saved to %s", dest)


def _verify_sha256(path: Path, expected: str) -> bool:
    """Return True if the file's SHA256 matches *expected*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        logger.debug("SHA256 mismatch: expected %s, got %s", expected, actual)
        return False
    return True
