"""Tests for runtime input-mode persistence."""

from __future__ import annotations

import pytest

from gazecontrol.runtime.input_mode import (
    InputMode,
    load_persisted_mode,
    persist_mode,
)


def test_input_mode_values():
    assert InputMode.HAND_ONLY.value == "hand"
    assert InputMode.EYE_HAND.value == "eye-hand"


def test_persist_and_load_roundtrip(tmp_path):
    target = tmp_path / "runtime.toml"
    persist_mode(InputMode.EYE_HAND, path=target)
    assert target.exists()
    loaded = load_persisted_mode(target)
    assert loaded == InputMode.EYE_HAND


def test_persist_and_load_hand_only(tmp_path):
    target = tmp_path / "runtime.toml"
    persist_mode(InputMode.HAND_ONLY, path=target)
    assert load_persisted_mode(target) == InputMode.HAND_ONLY


def test_load_returns_none_when_missing(tmp_path):
    assert load_persisted_mode(tmp_path / "absent.toml") is None


def test_load_returns_none_on_unknown_value(tmp_path):
    target = tmp_path / "runtime.toml"
    target.write_text('last_chosen_mode = "no-such-mode"\n', encoding="utf-8")
    assert load_persisted_mode(target) is None


def test_load_returns_none_on_corrupt_file(tmp_path):
    target = tmp_path / "runtime.toml"
    target.write_text("not [valid toml(((", encoding="utf-8")
    assert load_persisted_mode(target) is None


def test_persist_writes_show_selector_flag(tmp_path):
    target = tmp_path / "runtime.toml"
    persist_mode(InputMode.HAND_ONLY, show_selector_next_time=False, path=target)
    body = target.read_text(encoding="utf-8")
    assert 'last_chosen_mode = "hand"' in body
    assert "show_mode_selector = false" in body


@pytest.mark.parametrize("mode", list(InputMode))
def test_all_modes_roundtrip(tmp_path, mode):
    target = tmp_path / "runtime.toml"
    persist_mode(mode, path=target)
    assert load_persisted_mode(target) == mode


def test_persist_is_atomic_keeps_old_on_failure(tmp_path, monkeypatch):
    """Regression for BUG-005: a write that fails mid-flight must not leave
    a corrupt or empty runtime.toml.  The previous content must survive."""
    target = tmp_path / "runtime.toml"
    persist_mode(InputMode.HAND_ONLY, path=target)
    original = target.read_text(encoding="utf-8")

    # Force the staging write to raise — the final replace must NOT happen.
    real_write_text = type(target).write_text

    def boom(self, *a, **kw):
        if self.suffix == ".tmp":
            raise OSError("disk full simulation")
        return real_write_text(self, *a, **kw)

    monkeypatch.setattr(type(target), "write_text", boom)
    persist_mode(InputMode.EYE_HAND, path=target)  # swallows OSError + logs

    # Original file must still be readable and contain the prior mode.
    assert target.read_text(encoding="utf-8") == original
    # No leftover .tmp file.
    assert not (tmp_path / "runtime.toml.tmp").exists()
