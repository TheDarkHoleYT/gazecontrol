"""
OverlayWindow - Finestra trasparente always-on-top per il feedback HUD.
DEVE essere creata nel main thread Qt.

Thread-safety
-------------
The pipeline runs in a background thread.  Direct QWidget method calls from a
non-Qt thread are undefined behaviour in Qt.

Safe cross-thread update path (ARCH-6 fix):
  - ``_OverlayWidget`` declares a ``data_changed`` signal that carries the new
    HUD state dict.
  - The signal is connected to ``_on_data_changed`` with ``QueuedConnection``.
  - When ``OverlayWindow.update()`` emits the signal from any thread, Qt
    automatically marshals the call to the widget's owning (main) thread.
  - ``_on_data_changed`` stores the dict and schedules a repaint — both happen
    in the main thread, so no locking is needed at all.
"""

try:
    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QPainter
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from .hud_renderer import HUDRenderer


class _OverlayWidget(QWidget if HAS_PYQT else object):
    # Signal emitted from any thread; received in the Qt main thread.
    data_changed = pyqtSignal(dict) if HAS_PYQT else None  # type: ignore[assignment]

    def __init__(self):
        if not HAS_PYQT:
            return
        super().__init__()
        self.renderer = HUDRenderer()
        self._data: dict = {}
        # Connect with QueuedConnection so cross-thread emission is safe.
        self.data_changed.connect(
            self._on_data_changed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._setup_window()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(33)  # ~30fps

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.showFullScreen()
        self._apply_win32_clickthrough()

    def _apply_win32_clickthrough(self):
        """Imposta WS_EX_LAYERED | WS_EX_TRANSPARENT via Win32 API.
        Rende la finestra completamente invisibile agli input del mouse a livello OS.
        """
        try:
            import ctypes
            hwnd = int(self.winId())
            GWL_EXSTYLE       = -20
            WS_EX_LAYERED     = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_NOACTIVATE  = 0x08000000
            user32 = ctypes.windll.user32
            ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE,
                ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE,
            )
        except Exception:
            pass  # non Windows o hwnd non disponibile

    def _on_data_changed(self, data: dict) -> None:
        """Slot — always called in the Qt main thread (QueuedConnection)."""
        self._data = data
        # update() is a QWidget method; calling it here is safe (main thread).
        self.update()

    def paintEvent(self, event):
        # _data is only ever written by _on_data_changed (main thread) and read
        # here (also main thread) — no locking needed.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.renderer.render(painter, self._data)
        painter.end()


class OverlayWindow:
    """
    Manages the HUD widget.  ``create_widget()`` must be called from the Qt
    main thread, after ``QApplication`` has been created.

    ``update()`` is safe to call from any thread.
    """

    def __init__(self):
        self._widget: _OverlayWidget | None = None

    def create_widget(self) -> None:
        """Create the widget — call only from the Qt main thread."""
        if not HAS_PYQT:
            return
        self._widget = _OverlayWidget()

    def update(self, gaze_point, state, target_window, gesture_id, gesture_confidence,
               is_calibrated=False, gaze_event_type=None) -> None:
        """Thread-safe HUD update. Emits data_changed signal via QueuedConnection."""
        if self._widget is None:
            return
        self._widget.data_changed.emit({
            'gaze_point': gaze_point,
            'state': state,
            'target_window': target_window,
            'gesture_id': gesture_id,
            'gesture_confidence': gesture_confidence,
            'is_calibrated': is_calibrated,
            'gaze_event_type': gaze_event_type,
        })

    def stop(self) -> None:
        if self._widget:
            self._widget.close()
            self._widget = None
