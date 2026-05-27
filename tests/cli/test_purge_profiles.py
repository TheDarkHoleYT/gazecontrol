"""Tests for the GDPR-friendly --purge-profiles command (G16)."""

from __future__ import annotations

import json
import logging

from gazecontrol.cli import _cmd_purge_profiles
from gazecontrol.paths import Paths


def _populate(tmp_path):
    """Seed the profiles dir + runtime config so the purge has work to do."""
    profiles_dir = tmp_path / "profiles"
    runtime_cfg = tmp_path / "runtime.toml"
    (profiles_dir / "default" / "primary-legacy").mkdir(parents=True)
    (profiles_dir / "default" / "primary-legacy" / "v1.npz").write_bytes(b"x")
    runtime_cfg.write_text("input_mode = 'hand'\n", encoding="utf-8")
    return profiles_dir, runtime_cfg


def _patch_paths(monkeypatch, tmp_path):
    profiles_dir = tmp_path / "profiles"
    runtime_cfg = tmp_path / "runtime.toml"
    monkeypatch.setattr(Paths, "profiles", staticmethod(lambda **_: profiles_dir))
    monkeypatch.setattr(Paths, "runtime_config", staticmethod(lambda **_: runtime_cfg))
    return profiles_dir, runtime_cfg


def test_yes_flag_deletes_profiles_and_runtime_config(tmp_path, monkeypatch, capsys):
    profiles_dir, runtime_cfg = _patch_paths(monkeypatch, tmp_path)
    _populate(tmp_path)

    rc = _cmd_purge_profiles(assume_yes=True, as_json=False)

    assert rc == 0
    assert not profiles_dir.exists()
    assert not runtime_cfg.exists()


def test_no_targets_still_returns_zero(tmp_path, monkeypatch):
    """A fresh install (no profiles, no runtime.toml) must purge cleanly."""
    _patch_paths(monkeypatch, tmp_path)
    rc = _cmd_purge_profiles(assume_yes=True, as_json=False)
    assert rc == 0


def test_json_output_shape(tmp_path, monkeypatch, capsys):
    profiles_dir, runtime_cfg = _patch_paths(monkeypatch, tmp_path)
    _populate(tmp_path)

    rc = _cmd_purge_profiles(assume_yes=True, as_json=True)

    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert set(payload) == {"deleted", "paths"}
    assert payload["deleted"]["profiles_dir"] is True
    assert payload["deleted"]["runtime_config"] is True
    assert payload["paths"]["profiles_dir"] == str(profiles_dir)
    assert payload["paths"]["runtime_config"] == str(runtime_cfg)


def test_refuses_without_yes_when_stdin_not_a_tty(tmp_path, monkeypatch, capsys):
    """CI / scripts must explicitly opt in with --yes; the prompt path
    refuses when stdin is not a TTY."""
    _patch_paths(monkeypatch, tmp_path)
    _populate(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    rc = _cmd_purge_profiles(assume_yes=False, as_json=False)

    assert rc == 2
    err = capsys.readouterr().err
    assert "--yes" in err


def test_compliance_purge_event_logged(tmp_path, monkeypatch, caplog):
    """An audit trail must record the deletion regardless of payload."""
    _patch_paths(monkeypatch, tmp_path)
    _populate(tmp_path)

    with caplog.at_level(logging.INFO, logger="gazecontrol.compliance"):
        _cmd_purge_profiles(assume_yes=True, as_json=False)

    events = [r for r in caplog.records if r.message == "compliance.purge"]
    assert len(events) == 1
    # The extras travel via the LogRecord attributes (json formatter target).
    assert hasattr(events[0], "deleted")
    assert events[0].deleted["profiles_dir"] is True
