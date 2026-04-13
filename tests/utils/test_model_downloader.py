"""Tests for model_downloader — atomic download, SHA256, error handling."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gazecontrol.utils.model_downloader import _verify_sha256, ensure_model


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
        dest = tmp_path / "hand_landmarker.task"
        dest.write_bytes(b"mock model data")

        with patch("urllib.request.urlopen") as mock_urlopen:
            result = ensure_model("hand_landmarker.task", str(tmp_path))
            mock_urlopen.assert_not_called()

        assert result == str(dest)

    def test_download_writes_atomically(self, tmp_path):
        """File must appear as dest, not dest.part, after success."""
        content = _make_content(512)

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read = MagicMock(side_effect=[content, b""])

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = ensure_model("hand_landmarker.task", str(tmp_path))

        assert Path(result).exists()
        assert not Path(result + ".part").exists()
        assert Path(result).read_bytes() == content

    def test_download_failure_cleans_part_file(self, tmp_path):
        """On download failure the .part temp file must be removed."""
        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            with pytest.raises(RuntimeError, match="Failed to download"):
                ensure_model("hand_landmarker.task", str(tmp_path))

        part = tmp_path / "hand_landmarker.task.part"
        assert not part.exists()

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
                with pytest.raises(RuntimeError, match="SHA256 mismatch"):
                    ensure_model("hand_landmarker.task", str(tmp_path))
        finally:
            md._MODELS["hand_landmarker.task"] = orig
