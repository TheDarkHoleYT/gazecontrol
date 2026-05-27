"""Minimal i18n for GazeControl user-facing strings (G23).

The Qt HUD and the calibration runner have shipped Italian text since
v0.6 (Italian is Ciro's working language). For the v1.0 enterprise
release we add the *minimum* infrastructure to deliver English strings
to non-Italian operators without pulling in ``gettext`` / ``Babel``:

* A module-level :data:`LOCALE` ``dict[lang, dict[key, str]]`` keyed
  by two-letter ISO 639-1 codes (``"it"``, ``"en"``). Italian is the
  default; English is the only secondary locale today and is the
  fallback when a key is not yet translated.
* :func:`t(key)` resolves the active language from the
  ``GAZECONTROL_LOCALE`` env (preferred) or the system ``LANG``
  (fallback), then returns the translation — or the key itself if no
  translation exists in any locale (loud-but-safe behaviour: a
  missing string is obvious in the UI instead of silently rendering
  empty text).

Adding a language is a one-line dict insertion; adding a key is a
one-line addition to every locale dict. Both are caught by
``tests/test_i18n.py::test_all_locales_have_same_keys`` so
translations cannot drift.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_LANG: str = "it"
SUPPORTED_LANGS: tuple[str, ...] = ("it", "en")

#: ``LOCALE[lang][key]`` — translation strings. Keep keys hierarchical
#: (``"hud.hand_detected"``) so future namespaces stay readable.
LOCALE: Mapping[str, Mapping[str, str]] = {
    "it": {
        "hud.hand_detected": "● MANO RILEVATA",
        "hud.hand_waiting": "○ IN ATTESA MANO…",
        "hud.camera_error": "✗ ERRORE CAMERA",
        "calibration.progress": "Calibrazione gaze: {done}/{total}",
        "calibration.instructions": "Fissa il punto verde, tieni la testa ferma.",
        "compliance.purge.confirm_prompt": "Digita 'yes' per confermare: ",
        "compliance.purge.aborted": "Operazione annullata — nessun file eliminato.",
        "hud.backend_down": "GAZE OFFLINE — uso solo la mano",
        "hud.recenter_active": "Recenter in corso…",
    },
    "en": {
        "hud.hand_detected": "● HAND TRACKED",
        "hud.hand_waiting": "○ WAITING FOR HAND…",
        "hud.camera_error": "✗ CAMERA ERROR",
        "calibration.progress": "Gaze calibration: {done}/{total}",
        "calibration.instructions": "Fix the green dot, keep your head still.",
        "compliance.purge.confirm_prompt": "Type 'yes' to confirm: ",
        "compliance.purge.aborted": "Aborted — nothing deleted.",
        "hud.backend_down": "GAZE OFFLINE — hand only",
        "hud.recenter_active": "Recentering…",
    },
}


def active_lang() -> str:
    """Resolve the runtime language from env or default to Italian.

    Precedence:

    1. ``GAZECONTROL_LOCALE`` (full control: ``"it"``, ``"en"``).
    2. ``LANG`` (e.g. ``"en_US.UTF-8"`` → ``"en"``).
    3. :data:`DEFAULT_LANG` (Italian — the historical default).

    Unknown / unsupported locales fall through to ``DEFAULT_LANG`` so
    the UI never renders a missing-language placeholder.
    """
    explicit = os.environ.get("GAZECONTROL_LOCALE", "").strip().lower()
    if explicit and explicit in SUPPORTED_LANGS:
        return explicit
    lang_env = os.environ.get("LANG", "").strip().lower()
    if lang_env:
        prefix = lang_env.split("_", 1)[0].split(".", 1)[0]
        if prefix in SUPPORTED_LANGS:
            return prefix
    return DEFAULT_LANG


def t(key: str, **fmt: object) -> str:
    """Return the localized string for *key*, formatted with **fmt.

    Lookup order:

    1. Active language (see :func:`active_lang`).
    2. :data:`DEFAULT_LANG` fallback (Italian).
    3. The key itself — loud-but-safe so missing translations stay
       visible in the UI rather than rendering empty.

    Format placeholders use ``str.format`` syntax, e.g.::

        t("calibration.progress", done=3, total=13)
    """
    lang = active_lang()
    candidates = (lang, DEFAULT_LANG) if lang != DEFAULT_LANG else (DEFAULT_LANG,)
    for cand in candidates:
        bundle = LOCALE.get(cand, {})
        if key in bundle:
            template = bundle[key]
            try:
                return template.format(**fmt) if fmt else template
            except (KeyError, IndexError, ValueError):
                return template
    return key
