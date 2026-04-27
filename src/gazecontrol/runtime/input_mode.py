"""Input mode persistence — read/write the user's last chosen mode.

The mode-selector dialog calls :func:`persist_mode` whenever the user
confirms a choice with "remember choice" enabled. :func:`load_persisted_mode`
is consulted at startup so that returning users skip the dialog by default.

Storage format: a tiny TOML file under ``Paths.runtime_config()``::

    last_chosen_mode = "eye-hand"
    show_mode_selector = true

CLI ``--mode`` and the ``GAZECONTROL_RUNTIME__INPUT_MODE`` env variable
take precedence over the persisted file.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tomllib
from pathlib import Path

from gazecontrol.paths import Paths
from gazecontrol.settings import InputMode

logger = logging.getLogger(__name__)

__all__ = ["InputMode", "load_persisted_mode", "persist_mode"]


def load_persisted_mode(path: Path | None = None) -> InputMode | None:
    """Return the user's last chosen mode, or ``None`` if no file/invalid."""
    target = path or Paths.runtime_config()
    if not target.exists():
        return None
    try:
        with target.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        logger.warning("runtime config %s unreadable; ignoring.", target, exc_info=True)
        return None
    raw = data.get("last_chosen_mode")
    if raw is None:
        return None
    try:
        return InputMode(raw)
    except ValueError:
        logger.warning("runtime config: unknown mode %r; ignoring.", raw)
        return None


def persist_mode(
    mode: InputMode,
    *,
    show_selector_next_time: bool = True,
    path: Path | None = None,
) -> None:
    """Write the user's chosen mode to the runtime config file."""
    target = path or Paths.runtime_config()
    target.parent.mkdir(parents=True, exist_ok=True)
    body = (
        f'last_chosen_mode = "{mode.value}"\n'
        f"show_mode_selector = {str(show_selector_next_time).lower()}\n"
    )
    # Atomic write: never leave a half-written runtime.toml if the process
    # dies mid-write.  os.replace() is atomic on both POSIX and Windows.
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        logger.warning("Failed to persist runtime config to %s.", target, exc_info=True)
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
