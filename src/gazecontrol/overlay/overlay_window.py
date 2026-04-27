"""OverlayWindow — transparent always-on-top HUD for hand-control feedback.

MUST be created in the main Qt thread.

Thread-safety
-------------
The pipeline runs in a background thread.  Direct QWidget method calls from a
non-Qt thread are undefined behaviour in Qt.

Safe cross-thread update path:
  - ``_OverlayWidget`` declares a ``data_changed`` signal carrying the HUD dict.
  - The signal is connected with ``QueuedConnection``.
  - ``OverlayWindow.update()`` emits the signal from any thread; Qt marshals it
    to the widget's owning (main) thread automatically.
  - ``_on_data_changed`` stores the dict and schedules a repaint — both happen
    in the main thread, so no locking is needed.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from typing import TYPE_CHECKING, Any, cast

from PyQt6.QtCore import QMetaObject, Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QPainter, QPaintEvent
from PyQt6.QtWidgets import QWidget

from .hud_renderer import HudData, HUDRenderer
from .hud_state import HudState

if TYPE_CHECKING:
    from gazecontrol.window_manager.launcher import AppLauncher, LauncherApp

    from .launcher_panel import LauncherPanel

logger = logging.getLogger(__name__)

# PyQt6 is now a hard runtime dependency.  Kept as a module-level constant for
# backwards compatibility with test fixtures that historically toggled it.
HAS_PYQT: bool = True


class _OverlayWidget(QWidget):
    """Top-level transparent HUD widget. Receives ``HudData`` via ``data_changed``."""

    data_changed = pyqtSignal(dict)
    toggle_launcher_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.renderer = HUDRenderer()
        self._data: HudData = {}
        self._panel: LauncherPanel | None = None
        self.data_changed.connect(  # type: ignore[call-arg]
            self._on_data_changed,
            type=Qt.ConnectionType.QueuedConnection,
        )
        self.toggle_launcher_requested.connect(  # type: ignore[call-arg]
            self._on_toggle_launcher,
            type=Qt.ConnectionType.QueuedConnection,
        )
        self._setup_window()

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        try:
            primary = QGuiApplication.primaryScreen()
            if primary is not None:
                self.setGeometry(primary.geometry())
                wh = self.windowHandle()
                if wh is not None:
                    wh.setScreen(primary)
        except (RuntimeError, OSError) as exc:
            logger.debug("Overlay primary screen setup skipped: %s", exc)
        self.showFullScreen()
        self._apply_win32_clickthrough()

    def _apply_win32_clickthrough(self) -> None:
        """Set WS_EX_LAYERED | WS_EX_TRANSPARENT so clicks pass through."""
        try:
            hwnd = int(self.winId())
            GWLP_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_NOACTIVATE = 0x08000000
            user32 = ctypes.windll.user32
            user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
            user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
            ex_style = user32.GetWindowLongPtrW(hwnd, GWLP_EXSTYLE)
            user32.SetWindowLongPtrW(
                hwnd,
                GWLP_EXSTYLE,
                ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE,
            )
        except (OSError, AttributeError) as exc:
            # ctypes.windll only exists on Windows; non-Windows hosts skip silently.
            logger.debug("Overlay click-through setup skipped: %s", exc)

    def _on_data_changed(self, data: HudData) -> None:
        self._data = data
        self.update()

    def _on_toggle_launcher(self) -> None:
        """Toggle launcher panel visibility on the main Qt thread."""
        if self._panel is not None:
            self._panel.toggle()
            self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.renderer.render(painter, self._data)
        if self._panel is not None and self._panel.visible:
            self._panel.render(painter, self.width(), self.height())
        painter.end()


class OverlayWindow:
    """Manages the HUD widget.

    ``create_widget()`` must be called from the Qt main thread, after
    ``QApplication`` has been created.  ``update()`` is safe to call from any thread.
    """

    def __init__(self) -> None:
        self._widget: _OverlayWidget | None = None
        self._widget_lock = threading.Lock()

    def create_widget(self) -> None:
        """Create the widget — call only from the Qt main thread."""
        if not HAS_PYQT:
            return
        with self._widget_lock:
            self._widget = _OverlayWidget()

    def setup_launcher(self, apps: list[LauncherApp], launcher: AppLauncher) -> None:
        """Attach a :class:`~gazecontrol.overlay.launcher_panel.LauncherPanel`.

        Must be called from the Qt main thread, after :meth:`create_widget`.
        """
        if not HAS_PYQT or not apps:
            return
        from .launcher_panel import LauncherPanel

        panel = LauncherPanel(apps=apps, launcher=launcher)
        with self._widget_lock:
            widget = self._widget
        if widget is not None:
            widget._panel = panel

    def toggle_launcher(self) -> None:
        """Toggle the launcher panel visibility — safe to call from any thread."""
        with self._widget_lock:
            widget = self._widget
        if widget is not None:
            widget.toggle_launcher_requested.emit()

    def update(
        self,
        *,
        hud_state: HudState | None = None,
        fingertip_screen: tuple[int, int] | None = None,
        state: str = "IDLE",
        hovered_window: Any = None,
        gesture_id: str | None = None,
        gesture_confidence: float = 0.0,
        interaction_kind: str | None = None,
        launcher_visible: bool = False,
        capture_ok: bool = True,
        frame_bgr: Any = None,
        gaze_screen: tuple[int, int] | None = None,
        gaze_confidence: float = 0.0,
        pointer_source: str = "hand",
        input_mode: str = "hand",
    ) -> None:
        """Thread-safe HUD update.

        Pass *frame_bgr* (raw BGR ndarray at full capture resolution) — the
        renderer performs the thumbnail resize itself to avoid burdening the
        pipeline thread.

        Callers may pass a :class:`HudState` instance via *hud_state*
        (preferred) or individual keyword arguments.
        """
        if hud_state is not None:
            data: HudData = {
                "fingertip_screen": hud_state.fingertip_screen,
                "state": hud_state.state,
                "hovered_window": hud_state.hovered_window,
                "gesture_id": hud_state.gesture_id,
                "gesture_confidence": hud_state.gesture_confidence,
                "interaction_kind": hud_state.interaction_kind,
                "launcher_visible": hud_state.launcher_visible,
                "capture_ok": True,
                "frame_bgr": None,
                "gaze_screen": hud_state.gaze_screen,
                "gaze_confidence": hud_state.gaze_confidence,
                "pointer_source": hud_state.pointer_source,
                "input_mode": hud_state.input_mode,
            }
        else:
            data = {
                "fingertip_screen": fingertip_screen,
                "state": state,
                "hovered_window": hovered_window,
                "gesture_id": gesture_id,
                "gesture_confidence": gesture_confidence,
                "interaction_kind": interaction_kind,
                "launcher_visible": launcher_visible,
                "capture_ok": capture_ok,
                "frame_bgr": frame_bgr,
                "gaze_screen": gaze_screen,
                "gaze_confidence": gaze_confidence,
                "pointer_source": pointer_source,
                "input_mode": input_mode,
            }

        with self._widget_lock:
            widget = self._widget
        if widget is None:
            return
        widget.data_changed.emit(cast(dict[str, Any], data))

    def stop(self) -> None:
        """Close the overlay widget — safe to call from any thread."""
        with self._widget_lock:
            widget = self._widget
            self._widget = None
        if widget is not None:
            QMetaObject.invokeMethod(widget, "close", Qt.ConnectionType.QueuedConnection)
            widget.deleteLater()
