"""Foreground-application name lookup for per-app fusion overrides (G24).

The pipeline routes through :class:`PointerFusionStage`, which consults
``FusionSettings.app_overrides`` keyed by *exe name*. This module
implements the platform-specific name resolution behind a tiny pure
shim so the consumer stage stays portable: on Windows we walk
``GetForegroundWindow → GetWindowThreadProcessId → OpenProcess →
GetModuleBaseName``; everywhere else we return ``None`` and let the
pipeline fall through to the base fusion config.

The lookup is intentionally best-effort: every Win32 step is wrapped
in a broad except so a permission error, a sandboxed process, or a
race against window destruction can never raise into the pipeline.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def detect_foreground_app() -> str | None:
    """Return the foreground process' base name (lowercase), or ``None``.

    Examples (Windows): ``"firefox.exe"``, ``"figma.exe"``, ``"code.exe"``.

    Returns ``None`` on non-Windows platforms (until a portable
    implementation lands) or whenever any Win32 call fails — callers
    treat that as "no override applies for this frame".
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
    except (AttributeError, OSError):
        return None

    try:
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return None
        # PROCESS_QUERY_LIMITED_INFORMATION (0x1000) is enough for
        # GetModuleBaseName and works against high-integrity processes
        # without requiring admin elevation.
        handle = kernel32.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return None
        try:
            buf = ctypes.create_unicode_buffer(260)
            n = psapi.GetModuleBaseNameW(handle, None, buf, 260)
            if not n:
                return None
            return buf.value.lower()
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError, ValueError) as exc:
        logger.debug("detect_foreground_app: lookup failed: %s", exc)
        return None
