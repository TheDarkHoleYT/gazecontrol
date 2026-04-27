"""Tests for model_downloader — atomic download, SHA256, error handling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gazecontrol.errors import ModelDownloadError
from gazecontrol.utils.model_downloader import _verify_sha256, ensure_model

# Environment variable to bypass the pinned-SHA requirement in download tests.
_ALLOW_UNPINNED = {"GAZECONTROL_ALLOW_UNPINNED_MODELS": "1"}


def _make_content(n_bytes: int = 1024) -> bytes:
    return b"a" * n_bytes


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestVerifySha256:
    def test_correct_hash(self, tmp_path):
        data = _make_content()
        f = tmp_path / "file.bin"
        f.write_bytes(data)
        assert _verify_sha256(f, _sha256(data)) is True

    def test_wrong_hash(self, tmp_path):
        f = tmp_path / "file.bin"
        f.write_bytes(_make_content())
        assert _verify_sha256(f, "0" * 64) is False


class TestEnsureModel:
    def test_unknown_model_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown model"):
            ensure_model("nonexistent.task", str(tmp_path))

    def test_existing_file_returned_without_download(self, tmp_path):
        """A cached file that matches the pinned SHA is returned as-is."""
        from gazecontrol.utils import model_downloader as md

        content = b"mock model data"
        dest = tmp_path / "hand_landmarker.task"
        dest.write_bytes(content)

        # Pin the SHA of the mock content so the cache is treated as valid.
        orig = md._MODELS["hand_landmarker.task"]
        md._MODELS["hand_landmarker.task"] = (orig[0], _sha256(content))
        try:
            with patch("urllib.request.urlopen") as mock_urlopen:
                result = ensure_model("hand_landmarker.task", str(tmp_path))
                mock_urlopen.assert_not_called()
        finally:
            md._MODELS["hand_landmarker.task"] = orig

        assert result == str(dest)

    def test_download_writes_atomically(self, tmp_path):
        """File must appear as dest, not dest.part, after success."""
        from gazecontrol.utils import model_downloader as md

        content = _make_content(512)
        # Pin the SHA of the synthetic content so the post-download verify
        # passes, then the rename completes.
        orig = md._MODELS["hand_landmarker.task"]
        md._MODELS["hand_landmarker.task"] = (orig[0], _sha256(content))

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read = MagicMock(side_effect=[content, b""])

        try:
            with patch("urllib.request.urlopen", return_value=mock_response):
                result = ensure_model("hand_landmarker.task", str(tmp_path))
        finally:
            md._MODELS["hand_landmarker.task"] = orig

        assert Path(result).exists()
        assert not Path(result + ".part").exists()
        assert Path(result).read_bytes() == content

    def test_download_failure_cleans_part_file(self, tmp_path, monkeypatch):
        """On download failure the .part temp file must be removed."""
        monkeypatch.setenv("GAZECONTROL_ALLOW_UNPINNED_MODELS", "1")
        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            with pytest.raises(ModelDownloadError, match="Failed to download"):
                ensure_model("hand_landmarker.task", str(tmp_path))

        part = tmp_path / "hand_landmarker.task.part"
        assert not part.exists()

    def test_unpinned_download_refused_without_env(self, tmp_path):
        """Downloading a model with no pinned SHA must raise ValueError.

        All shipped models are now pinned, so we synthesise an unpinned entry
        in the registry to verify the gate still triggers.
        """
        from gazecontrol.utils import model_downloader as md

        orig = md._MODELS["hand_landmarker.task"]
        md._MODELS["hand_landmarker.task"] = (orig[0], None)
        try:
            with pytest.raises(ValueError, match="no pinned SHA256"):
                ensure_model("hand_landmarker.task", str(tmp_path))
        finally:
            md._MODELS["hand_landmarker.task"] = orig

    def test_http_url_rejected(self, tmp_path):
        """Model URLs that are not HTTPS must raise ValueError."""
        from gazecontrol.utils import model_downloader as md

        orig = md._MODELS["hand_landmarker.task"]
        md._MODELS["hand_landmarker.task"] = ("http://insecure.example.com/model.task", None)
        try:
            with pytest.raises(ValueError, match="does not use HTTPS"):
                ensure_model("hand_landmarker.task", str(tmp_path))
        finally:
            md._MODELS["hand_landmarker.task"] = orig

    def test_sha256_mismatch_for_cached_file_triggers_redownload(self, tmp_path, monkeypatch):
        """Cached file with wrong SHA is deleted and re-download attempted."""
        monkeypatch.setenv("GAZECONTROL_ALLOW_UNPINNED_MODELS", "1")
        dest = tmp_path / "hand_landmarker.task"
        dest.write_bytes(b"stale content")

        from gazecontrol.utils import model_downloader as md

        orig = md._MODELS["hand_landmarker.task"]
        # Register a non-None SHA that won't match the stale file.
        md._MODELS["hand_landmarker.task"] = (orig[0], "a" * 64)
        try:
            with patch("urllib.request.urlopen", side_effect=OSError("network")):
                with pytest.raises(ModelDownloadError, match="Failed to download"):
                    ensure_model("hand_landmarker.task", str(tmp_path))
            # Stale file must be deleted by the SHA mismatch path.
            assert not dest.exists()
        finally:
            md._MODELS["hand_landmarker.task"] = orig

    def test_part_file_cleaned_on_mid_write_failure(self, tmp_path, monkeypatch):
        """If the download fails mid-write, the .part file must be removed.

        ``_download`` retries up to 3 times — the mock returns a chunk then an
        OSError on every attempt so the part file is recreated and cleaned
        each retry.
        """
        monkeypatch.setenv("GAZECONTROL_ALLOW_UNPINNED_MODELS", "1")
        # Override the back-off so the test does not actually sleep.
        monkeypatch.setattr(
            "gazecontrol.utils.model_downloader._RETRY_BACKOFF_S",
            (0.0, 0.0, 0.0),
        )

        def _make_mock_response():
            r = MagicMock()
            r.__enter__ = lambda s: s
            r.__exit__ = MagicMock(return_value=False)
            r.read = MagicMock(side_effect=[b"initial chunk", OSError("io error")])
            return r

        with patch("urllib.request.urlopen", side_effect=lambda *a, **k: _make_mock_response()):
            with pytest.raises(ModelDownloadError):
                ensure_model("hand_landmarker.task", str(tmp_path))

        part = tmp_path / "hand_landmarker.task.part"
        assert not part.exists()

    def test_all_registered_models_have_pinned_sha(self):
        """Regression for BUG-003: every registered model must ship with a
        pinned SHA256.  Unpinned entries open a supply-chain hole because
        ``GAZECONTROL_ALLOW_UNPINNED_MODELS=1`` would skip integrity checks
        on download.  Re-introducing ``None`` requires updating this test
        AND the security note in :mod:`gazecontrol.utils.model_downloader`.
        """
        from gazecontrol.utils import model_downloader as md

        unpinned = [name for name, (_, sha) in md._MODELS.items() if sha is None]
        assert not unpinned, f"Unpinned models: {unpinned}"

    def test_sha256_mismatch_raises(self, tmp_path):
        """If a checksum is registered and mismatches, raise RuntimeError."""
        content = _make_content()

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read = MagicMock(side_effect=[content, b""])

        # Temporarily register a fake sha256 for the model.
        from gazecontrol.utils import model_downloader as md

        orig = md._MODELS["hand_landmarker.task"]
        md._MODELS["hand_landmarker.task"] = (orig[0], "0" * 64)
        try:
            with patch("urllib.request.urlopen", return_value=mock_response):
                with pytest.raises(ModelDownloadError, match="SHA256 mismatch"):
                    ensure_model("hand_landmarker.task", str(tmp_path))
        finally:
            md._MODELS["hand_landmarker.task"] = orig
