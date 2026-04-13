import math
import time

from gazecontrol.config import (
    OVERLAY_GAZE_DOT_RADIUS, OVERLAY_GAZE_DOT_COLOR,
    OVERLAY_TARGETING_COLOR, OVERLAY_READY_COLOR,
)

try:
    from PyQt6.QtCore import Qt, QRectF, QPointF
    from PyQt6.QtGui import QPen, QBrush, QColor, QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


class HUDRenderer:
    def __init__(self):
        self._start_time = time.time()

    def render(self, painter, data):
        if not HAS_PYQT:
            return

        gaze_point = data.get('gaze_point')
        state = data.get('state', 'IDLE')
        target_window = data.get('target_window')
        gesture_id = data.get('gesture_id')
        gesture_confidence = data.get('gesture_confidence', 0)

        self._draw_state_label(painter, state)

        is_calibrated = data.get('is_calibrated', False)
        if gaze_point:
            self._draw_gaze_dot(painter, gaze_point, is_calibrated)

        if target_window and state in ('TARGETING', 'READY', 'ACTING'):
            color = OVERLAY_READY_COLOR if state in ('READY', 'ACTING') else OVERLAY_TARGETING_COLOR
            thickness = 4 if state in ('READY', 'ACTING') else 3
            self._draw_window_border(painter, target_window, color, thickness)

        if gesture_id:
            self._draw_gesture_info(painter, gesture_id, gesture_confidence)

    def _draw_gaze_dot(self, painter, gaze_point, is_calibrated=False):
        elapsed = time.time() - self._start_time
        pulse = math.sin(elapsed * 4.0) * 3.0
        x, y = gaze_point

        if is_calibrated:
            # Cerchio pieno verde pulsante
            radius = OVERLAY_GAZE_DOT_RADIUS + pulse
            r, g, b = OVERLAY_GAZE_DOT_COLOR
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(r, g, b, 200)))
            painter.drawEllipse(QRectF(x - radius, y - radius, radius * 2, radius * 2))
        else:
            # Anello arancione tratteggiato + mirino — indica modalità non calibrata
            radius = OVERLAY_GAZE_DOT_RADIUS + 4 + pulse
            pen = QPen(QColor(255, 160, 0, 210), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(x - radius, y - radius, radius * 2, radius * 2))

            # Mirino (crosshair) centrale
            arm = radius * 0.6
            painter.setPen(QPen(QColor(255, 160, 0, 180), 1))
            painter.drawLine(QPointF(x - arm, y), QPointF(x + arm, y))
            painter.drawLine(QPointF(x, y - arm), QPointF(x, y + arm))

            # Etichetta "RAW" piccola sopra il cerchio
            font = QFont('Segoe UI', 7)
            painter.setFont(font)
            painter.setPen(QPen(QColor(255, 160, 0, 180)))
            painter.drawText(int(x - 10), int(y - radius - 4), 'RAW')

    def _draw_window_border(self, painter, target_window, color, thickness):
        rect = target_window.get('rect')
        if not rect:
            return
        x, y, w, h = rect
        r, g, b = color
        pen = QPen(QColor(r, g, b, 220), thickness)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(x, y, w, h))

    def _draw_gesture_info(self, painter, gesture_id, confidence):
        painter.setPen(QPen(QColor(255, 255, 255, 230)))
        font = QFont('Segoe UI', 12, QFont.Weight.Bold)
        painter.setFont(font)

        screen_rect = painter.window()
        x = screen_rect.width() - 200
        y = screen_rect.height() - 80

        painter.drawText(x, y, gesture_id)

        bar_y = y + 10
        bar_w = 160
        bar_h = 12
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(60, 60, 60, 180)))
        painter.drawRect(QRectF(x, bar_y, bar_w, bar_h))

        conf = max(0.0, min(1.0, confidence or 0))
        g = int(conf * 200)
        r = int((1 - conf) * 200)
        painter.setBrush(QBrush(QColor(r, g, 50, 220)))
        painter.drawRect(QRectF(x, bar_y, bar_w * conf, bar_h))

    def _draw_state_label(self, painter, state):
        font = QFont('Segoe UI', 9)
        painter.setFont(font)
        painter.setPen(QPen(QColor(200, 200, 200, 200)))
        painter.drawText(10, 20, f'State: {state}')
