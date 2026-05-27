r"""Stable monitor identifier for per-output profile scoping (G21).

Multi-monitor desktops (laptop + external) need per-display gaze
calibration: a profile fit on a 1080p external is wrong on a 4K laptop
panel because the angle → pixel mapping changes with the physical
geometry. ADR-0009 chose ``<profiles>/<user>/<monitor>/v{N}.npz`` as
the on-disk layout; this module produces the ``monitor`` segment.

We deliberately avoid OS-specific APIs (EDID parsing, Win32 device
instance IDs) so the implementation is portable and unit-testable.
The id is the SHA1-truncated hash of:

* the screen *name* (``QGuiApplication.screens()[i].name()``, e.g.
  ``"\\\\.\\DISPLAY1"`` on Windows or ``"HDMI-1"`` on X11), and
* the screen *geometry* ``(x, y, width, height)``.

When the name changes after reseating a cable but the geometry stays
the same, the id changes — that is the correct outcome because the
mapping between angles and physical pixels may also have changed.

When the geometry changes (e.g. the user moves a window from a 4K
panel to a 1080p panel without changing screen positions), the id
also changes — likewise correct.

The default ``DEFAULT_MONITOR_ID`` (``"primary-legacy"``) is the
sentinel used by the Phase-0 profile migrator; it keeps the runtime
working when Qt is unavailable (CLI dump-config, headless tests).
"""

from __future__ import annotations

import hashlib
import re

DEFAULT_MONITOR_ID: str = "primary-legacy"

#: Length of the hex digest preserved in the id. 12 hex chars ≈ 48 bits
#: of entropy — plenty for the handful of monitors a typical desktop
#: ever sees while keeping the on-disk directory names short.
_HEX_LEN: int = 12

#: Maximum length of the sanitised name prefix in the id. Long enough
#: to recognise the monitor at a glance (``"HDMI-1__a3f5…"``); short
#: enough not to blow past common path-length limits on Windows.
_NAME_PREFIX_LEN: int = 16

_SAFE_CHAR_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def monitor_id_from_screen_info(
    name: str,
    geometry: tuple[int, int, int, int],
) -> str:
    """Return a deterministic, filesystem-safe monitor id.

    Args:
        name:     Screen name reported by ``QGuiApplication.screens()[i].name()``.
                  May contain backslashes / dots / spaces on Windows.
        geometry: ``(x, y, width, height)`` in device-pixel coords.

    Returns:
        ``"<prefix>__<hash>"`` where *prefix* is the screen name with
        unsafe characters replaced by ``-``, truncated to
        :data:`_NAME_PREFIX_LEN`, and *hash* is the first
        :data:`_HEX_LEN` hex chars of the SHA1 of
        ``name|geometry``. Two screens with the same name but
        different geometry get different ids; two screens with
        different names but the same geometry also get different ids.
    """
    safe = _SAFE_CHAR_RE.sub("-", name or "screen").strip("-_") or "screen"
    prefix = safe[:_NAME_PREFIX_LEN].rstrip("-_") or "screen"
    payload = f"{name}|{geometry[0]}x{geometry[1]}+{geometry[2]}+{geometry[3]}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:_HEX_LEN]  # noqa: S324
    return f"{prefix}__{digest}"


def detect_active_monitor_id(default: str = DEFAULT_MONITOR_ID) -> str:
    """Return the monitor id of the *primary* Qt screen, or *default*.

    A thin convenience around :func:`monitor_id_from_screen_info` that
    queries ``QGuiApplication.primaryScreen()``. When Qt is not
    available (headless CI, CLI ``--dump-config``) we fall back to
    *default* so the rest of the runtime keeps working.
    """
    try:
        from PyQt6.QtGui import QGuiApplication
    except ImportError:
        return default
    try:
        screen = QGuiApplication.primaryScreen()
    except RuntimeError:
        # QGuiApplication exists but no instance yet — typical in
        # ``--dump-config`` flows that never start a Qt app.
        return default
    if screen is None:
        return default
    geo = screen.geometry()
    return monitor_id_from_screen_info(
        screen.name(),
        (int(geo.x()), int(geo.y()), int(geo.width()), int(geo.height())),
    )
