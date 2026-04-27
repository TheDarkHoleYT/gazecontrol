"""Model downloader — fetch MediaPipe Tasks assets with safety checks.

Security policy:
- All registered models MUST have a pinned SHA256.  Downloading a model
  without a pinned checksum is refused unless the environment variable
  ``GAZECONTROL_ALLOW_UNPINNED_MODELS=1`` is set.  This protects against
  MITM / CDN compromise delivering arbitrary TFLite payloads into MediaPipe's
  native execution path.
- Only HTTPS URLs are accepted.
- Downloads are atomic: written to ``<dest>.part`` then renamed on success.
- Partial files are removed on any failure.

To update checksums after a new model release:
    python -c "import hashlib,pathlib; \
        print(hashlib.sha256(pathlib.Path('model.tflite').read_bytes()).hexdigest())"
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from gazecontrol.errors import ModelDownloadError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

#: Map from filename → (URL, sha256_hex | None)
#: sha256 values must be kept up-to-date when models are re-released upstream.
#: Compute with: sha256sum <file>  or  certutil -hashfile <file> SHA256 (Windows).
_MODELS: dict[str, tuple[str, str | None]] = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_short_range/float16/1/"
        "blaze_face_short_range.tflite",
        "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    ),
}

# Per-attempt timeout (seconds).  Increases linearly across retries so
# slow connections still complete on the second/third attempt.
_DOWNLOAD_TIMEOUTS: tuple[int, ...] = (30, 60, 120)
_RETRY_BACKOFF_S: tuple[float, ...] = (0.0, 2.0, 8.0)

# Environment flag to allow downloading models without a pinned SHA256.
_ALLOW_UNPINNED_ENV = "GAZECONTROL_ALLOW_UNPINNED_MODELS"


def ensure_model(model_name: str, models_dir: str | os.PathLike[str]) -> str:
    """Return the local path of *model_name*, downloading if absent.

    Args:
        model_name: File name registered in ``_MODELS``.
        models_dir: Directory where the model should be stored.

    Returns:
        Absolute path to the downloaded model file.

    Raises:
        ValueError:    Model name not in registry, or unpinned SHA256 and
                       ``GAZECONTROL_ALLOW_UNPINNED_MODELS`` not set.
        RuntimeError:  Download failed or SHA256 mismatch after download.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / model_name

    entry = _MODELS.get(model_name)
    if entry is None:
        raise ValueError(f"Unknown model: {model_name!r}. Registered models: {list(_MODELS)}")
    url, expected_sha256 = entry

    # Enforce HTTPS scheme.
    if not url.startswith("https://"):
        raise ValueError(
            f"Model URL for {model_name!r} does not use HTTPS: {url!r}. "
            "Only HTTPS model URLs are accepted for security reasons."
        )

    if dest.exists():
        if expected_sha256 and not _verify_sha256(dest, expected_sha256):
            logger.warning("SHA256 mismatch for cached %s — re-downloading.", model_name)
            dest.unlink()
        else:
            return str(dest)

    # File not present (or just invalidated) — must download.
    # Refuse to download a model without a pinned checksum unless the user opts in.
    # Already-cached files are returned above regardless of checksum availability.
    if expected_sha256 is None and not _allow_unpinned():
        raise ValueError(
            f"Model {model_name!r} has no pinned SHA256 checksum. "
            f"Set {_ALLOW_UNPINNED_ENV}=1 to allow downloading unverified models. "
            "WARNING: this bypasses integrity verification — use only in trusted environments."
        )

    _download(model_name, url, dest, expected_sha256)
    return str(dest)


def _allow_unpinned() -> bool:
    """Return True when the user has opted into downloading unpinned models."""
    return os.environ.get(_ALLOW_UNPINNED_ENV, "0").strip() == "1"


def _download(
    model_name: str,
    url: str,
    dest: Path,
    expected_sha256: str | None,
) -> None:
    """Download *url* to *dest* with retry, atomic rename, and SHA256 verify.

    Retries up to ``len(_DOWNLOAD_TIMEOUTS)`` times with exponential back-off
    on transient network failures (URLError, OSError).  A SHA256 mismatch is
    treated as a hard failure: not retried (the upstream URL has rotated).
    """
    part = dest.with_suffix(dest.suffix + ".part")

    last_error: Exception | None = None
    for attempt, (timeout, backoff) in enumerate(
        zip(_DOWNLOAD_TIMEOUTS, _RETRY_BACKOFF_S, strict=True), start=1
    ):
        if backoff > 0:
            logger.info(
                "Retrying %s download (attempt %d/%d) after %.1fs backoff",
                model_name,
                attempt,
                len(_DOWNLOAD_TIMEOUTS),
                backoff,
            )
            time.sleep(backoff)
        logger.info("Downloading %s from %s (timeout=%ds)", model_name, url, timeout)
        try:
            with (
                urllib.request.urlopen(  # noqa: S310 (URL validated as HTTPS above)
                    url, timeout=timeout
                ) as response,
                part.open("wb") as fh,
            ):
                while chunk := response.read(1 << 20):  # 1 MB chunks
                    fh.write(chunk)
            break
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            logger.warning("Download attempt %d failed: %s", attempt, exc)
            if part.exists():
                part.unlink()
    else:
        raise ModelDownloadError(
            f"Failed to download {model_name} after "
            f"{len(_DOWNLOAD_TIMEOUTS)} attempts: {last_error}"
        )

    if expected_sha256 and not _verify_sha256(part, expected_sha256):
        part.unlink()
        raise ModelDownloadError(
            f"SHA256 mismatch for {model_name} after download. "
            "The file may be corrupted or the URL has changed."
        )

    # Atomic rename — avoids leaving a half-written file if the process dies.
    os.replace(part, dest)
    logger.info("Model saved: %s", dest.name)


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
