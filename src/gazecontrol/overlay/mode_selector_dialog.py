"""ModeSelectorDialog — startup chooser between HAND_ONLY and EYE_HAND.

Shown by ``cli.main`` before the pipeline thread starts. Two large
buttons select the mode; a small footer offers the calibration shortcut
and the doctor command. The user choice is exposed via :attr:`result_mode`
after :meth:`exec` returns.
"""

from __future__ import annotations

from typing import Any

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )

    HAS_PYQT = True
except ImportError:  # pragma: no cover
    HAS_PYQT = False

from gazecontrol.runtime.input_mode import InputMode

_CARD_STYLE = """
QPushButton {
    background-color: rgba(40, 44, 60, 220);
    color: white;
    border: 2px solid rgba(120, 140, 200, 200);
    border-radius: 14px;
    padding: 32px 24px;
    font-size: 18px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: rgba(60, 80, 140, 240);
    border: 2px solid rgba(180, 220, 255, 240);
}
QPushButton:focus {
    outline: 3px solid rgba(0, 220, 100, 200);
}
"""


class ModeSelectorDialog(QDialog if HAS_PYQT else object):  # type: ignore[misc]
    """Startup mode selector. Use :meth:`exec` and read :attr:`result_mode`."""

    def __init__(self, initial: Any = InputMode.HAND_ONLY, parent: Any = None) -> None:
        self.result_mode: InputMode | None = None
        if not HAS_PYQT:
            return
        super().__init__(parent)
        self.setWindowTitle("GazeControl — Input Mode")
        self.setModal(True)
        self.setStyleSheet("QDialog { background-color: rgba(20, 22, 30, 245); }")
        self.setMinimumSize(620, 360)

        self._initial = self._coerce_mode(initial)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        title = QLabel("How do you want to control your desktop?")
        title.setStyleSheet("color: white; font-size: 20px; font-weight: 600;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        cards = QHBoxLayout()
        cards.setSpacing(20)

        self._btn_hand = QPushButton("✋\nHand Only\n\nPoint, pinch, drag, resize.")
        self._btn_eye = QPushButton("👁️ + ✋\nEye + Hand\n\nLook to target, hand to act.")
        for btn in (self._btn_hand, self._btn_eye):
            btn.setStyleSheet(_CARD_STYLE)
            btn.setMinimumHeight(180)
        self._btn_hand.clicked.connect(lambda: self._on_choose(InputMode.HAND_ONLY))
        self._btn_eye.clicked.connect(lambda: self._on_choose(InputMode.EYE_HAND))
        cards.addWidget(self._btn_hand)
        cards.addWidget(self._btn_eye)

        self._remember_box = QCheckBox("Remember my choice")
        self._remember_box.setChecked(True)
        self._remember_box.setStyleSheet("color: rgba(220, 220, 220, 220);")

        footer = QHBoxLayout()
        footer.setSpacing(12)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        cancel.setStyleSheet(
            "QPushButton { color: rgba(220,220,220,220); padding: 6px 14px;"
            " background: transparent; border: 1px solid rgba(120,120,140,180);"
            " border-radius: 6px; }"
        )
        footer.addWidget(self._remember_box)
        footer.addStretch(1)
        footer.addWidget(cancel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 28, 36, 22)
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addLayout(cards)
        layout.addStretch(1)
        layout.addLayout(footer)

        # Pre-select the recommended button (initial mode).
        if self._initial == InputMode.EYE_HAND:
            self._btn_eye.setFocus()
        else:
            self._btn_hand.setFocus()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_choose(self, mode: InputMode) -> None:
        self.result_mode = mode
        self._remember = self._remember_box.isChecked()
        self.accept()

    @property
    def remember(self) -> bool:
        """True when the user wants the choice persisted."""
        return getattr(self, "_remember", True)

    @staticmethod
    def _coerce_mode(value: Any) -> InputMode:
        if isinstance(value, InputMode):
            return value
        try:
            return InputMode(value)
        except (ValueError, TypeError):
            return InputMode.HAND_ONLY


def show_mode_selector(initial: InputMode = InputMode.HAND_ONLY) -> InputMode | None:
    """Convenience helper used by ``cli.main`` and integration tests."""
    if not HAS_PYQT:
        return None
    app = QApplication.instance() or QApplication([])  # noqa: F841
    dialog = ModeSelectorDialog(initial=initial)
    dialog.exec()
    return dialog.result_mode
