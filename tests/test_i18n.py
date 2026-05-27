"""Tests for the minimal i18n helper (G23)."""

from __future__ import annotations

import pytest

from gazecontrol.i18n import (
    DEFAULT_LANG,
    LOCALE,
    SUPPORTED_LANGS,
    active_lang,
    t,
)

# ---------------------------------------------------------------------------
# Translation contract — invariants every locale must satisfy
# ---------------------------------------------------------------------------


def test_all_supported_languages_have_a_locale_entry():
    for lang in SUPPORTED_LANGS:
        assert lang in LOCALE, f"missing LOCALE entry for {lang!r}"


def test_default_language_is_supported():
    assert DEFAULT_LANG in SUPPORTED_LANGS


def test_all_locales_have_the_same_keys():
    """Translation drift guard: every locale exposes the same key set.

    Adding a key to one locale without translating it in another shows
    up here, not in production when a user picks the missing language.
    """
    reference_keys = set(LOCALE[DEFAULT_LANG].keys())
    for lang, bundle in LOCALE.items():
        missing = reference_keys - set(bundle.keys())
        extra = set(bundle.keys()) - reference_keys
        assert not missing, f"{lang!r} missing keys: {sorted(missing)}"
        assert not extra, f"{lang!r} has extra keys: {sorted(extra)}"


# ---------------------------------------------------------------------------
# active_lang
# ---------------------------------------------------------------------------


def test_active_lang_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("GAZECONTROL_LOCALE", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    assert active_lang() == DEFAULT_LANG


def test_active_lang_gazecontrol_locale_wins_over_LANG(monkeypatch):
    monkeypatch.setenv("GAZECONTROL_LOCALE", "en")
    monkeypatch.setenv("LANG", "it_IT.UTF-8")
    assert active_lang() == "en"


def test_active_lang_falls_back_to_lang_env(monkeypatch):
    monkeypatch.delenv("GAZECONTROL_LOCALE", raising=False)
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert active_lang() == "en"


def test_active_lang_unknown_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("GAZECONTROL_LOCALE", "klingon")
    monkeypatch.delenv("LANG", raising=False)
    assert active_lang() == DEFAULT_LANG


def test_active_lang_strips_lang_encoding(monkeypatch):
    """LANG=en_GB.UTF-8 → en."""
    monkeypatch.delenv("GAZECONTROL_LOCALE", raising=False)
    monkeypatch.setenv("LANG", "en_GB.UTF-8")
    assert active_lang() == "en"


# ---------------------------------------------------------------------------
# t
# ---------------------------------------------------------------------------


def test_t_returns_active_language_translation(monkeypatch):
    monkeypatch.setenv("GAZECONTROL_LOCALE", "en")
    monkeypatch.delenv("LANG", raising=False)
    assert t("hud.hand_detected") == "● HAND TRACKED"


def test_t_default_language_translation(monkeypatch):
    monkeypatch.delenv("GAZECONTROL_LOCALE", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    assert t("hud.hand_detected") == "● MANO RILEVATA"


def test_t_supports_format_placeholders(monkeypatch):
    monkeypatch.setenv("GAZECONTROL_LOCALE", "en")
    monkeypatch.delenv("LANG", raising=False)
    assert t("calibration.progress", done=3, total=13) == "Gaze calibration: 3/13"


def test_t_missing_key_returns_key_unchanged(monkeypatch):
    monkeypatch.setenv("GAZECONTROL_LOCALE", "en")
    monkeypatch.delenv("LANG", raising=False)
    assert t("not.a.real.key") == "not.a.real.key"


def test_t_falls_back_to_default_when_key_missing_in_active_lang(monkeypatch):
    """If we ever forget to translate a key, the default-language
    string surfaces rather than the bare key."""
    # Temporarily strip a key from the "en" bundle to simulate the
    # drift case. (LOCALE is a Mapping but the inner dicts are mutable.)
    monkeypatch.setenv("GAZECONTROL_LOCALE", "en")
    monkeypatch.delenv("LANG", raising=False)
    original = LOCALE["en"]["hud.recenter_active"]
    try:
        del LOCALE["en"]["hud.recenter_active"]
        assert t("hud.recenter_active") == LOCALE[DEFAULT_LANG]["hud.recenter_active"]
    finally:
        LOCALE["en"]["hud.recenter_active"] = original


def test_t_handles_missing_format_keys_without_raising(monkeypatch):
    """A template that references a placeholder the caller forgot to
    pass must surface the raw template instead of raising."""
    monkeypatch.setenv("GAZECONTROL_LOCALE", "en")
    # calibration.progress expects {done} and {total}. Pass neither.
    assert t("calibration.progress") == LOCALE["en"]["calibration.progress"]


@pytest.mark.parametrize("lang", SUPPORTED_LANGS)
def test_t_for_every_supported_language_returns_non_empty(lang, monkeypatch):
    """Sanity: every key in every locale renders a non-empty string."""
    monkeypatch.setenv("GAZECONTROL_LOCALE", lang)
    for key in LOCALE[DEFAULT_LANG]:
        rendered = t(key)
        assert rendered, f"{lang!r}: key {key!r} renders empty"
