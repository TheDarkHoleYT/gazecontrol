"""AppLauncher — subprocess-based application launcher for the hand gesture panel."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Windows-specific: CREATE_NEW_PROCESS_GROUP ensures the child process is fully
# detached from GazeControl and continues running even if GazeControl exits.
_DETACH_FLAGS = (
    subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined,unused-ignore]
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP")
    else 0
)


@dataclass(frozen=True)
class LauncherApp:
    """Configuration for a single launchable application.

    Attributes:
        name: Display name shown in the launcher panel.
        exe:  Full path or command available on ``PATH``.
        args: Optional list of command-line arguments.
        icon: Optional path to an icon file (used by the launcher panel UI).
    """

    name: str
    exe: str
    args: tuple[str, ...] = field(default_factory=tuple)
    icon: str | None = None


class AppLauncher:
    """Launch applications via ``subprocess.Popen``.

    Each call to :meth:`launch` fires the application in a detached process;
    GazeControl does not wait for it to finish and continues running normally.
    """

    def launch(self, app: LauncherApp) -> None:
        """Start *app* in a new detached process.

        Failures are logged as warnings rather than raised so a bad entry in
        the launcher config does not crash the pipeline.

        Args:
            app: The :class:`LauncherApp` to launch.
        """
        cmd = [app.exe, *app.args]
        try:
            subprocess.Popen(
                cmd,
                creationflags=_DETACH_FLAGS,
                close_fds=True,
            )
            logger.info("AppLauncher: launched %r (cmd=%s)", app.name, cmd)
        except FileNotFoundError:
            logger.warning("AppLauncher: executable not found — %r (cmd=%s)", app.name, cmd)
        except Exception:
            logger.warning("AppLauncher: failed to launch %r", app.name, exc_info=True)
