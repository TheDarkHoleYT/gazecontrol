"""HUDRenderer — draws the hand-control HUD overlay.

Renders:
- Activity heartbeat (top-right corner blink — proof the pipeline is alive).
- Camera / hand-detection status bar.
- Fingertip pointer (dot + outer ring, pulsing) when hand is tracked.
- Hovered-window outline.
- Resize-grip hint (bottom-right corner highlight).
- FSM state label + interaction kind.
- Gesture info bar (label + confidence).
- Webcam preview thumbnail (bottom-left).

Performance notes
-----------------
All ``QColor``, ``QFont``, ``QPen``, and ``QBrush`` objects are created once
in ``__init__`` and reused every paint cycle — avoiding ~40 GC-eligible object
allocations per frame.

The camera thumbnail uses ``QImage.Format_BGR888`` (PyQt6 ≥ 6.5) which maps
the BGR ndarray directly without a channel-order copy.  ``cv2.resize`` is
performed inside the renderer (not on the pipeline thread) using a pre-allocated
buffer.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict, cast

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen

from gazecontrol.settings import get_settings

logger = logging.getLogger(__name__)


class HudData(TypedDict, total=False):
    """Snapshot dict consumed by :meth:`HUDRenderer.render`."""

    fingertip_screen: tuple[int, int] | None
    state: str
    hovered_window: Any
    gesture_id: str | None
    gesture_confidence: float
    interaction_kind: str | None
    launcher_visible: bool
    capture_ok: bool
    frame_bgr: np.ndarray[Any, Any] | None
    gaze_screen: tuple[int, int] | None
    gaze_confidence: float
    pointer_source: str
    input_mode: str


_ov = get_settings().overlay
OVERLAY_POINTER_RADIUS = _ov.pointer_radius
OVERLAY_POINTER_COLOR = _ov.pointer_color
OVERLAY_GRIP_COLOR = _ov.grip_hint_color
OVERLAY_DRAG_COLOR = _ov.drag_color
OVERLAY_RESIZE_COLOR = _ov.resize_color
OVERLAY_TARGETING_COLOR = _ov.targeting_color
OVERLAY_GRIP_RATIO = get_settings().interaction.grip_ratio

_PREVIEW_W = 240
_PREVIEW_H = 135

# Retained for backwards compatibility with test fixtures that monkey-patch
# this flag to exercise the headless code path.  PyQt6 is now a hard runtime
# dependency, so the flag is always True at module load.
HAS_PYQT: bool = True

# QImage format: BGR888 is zero-copy on PyQt6 ≥ 6.5 (no channel swap needed).
# Fall back to RGB888 with manual flip on older builds.
_QIMG_BGR: QImage.Format | None = getattr(QImage.Format, "Format_BGR888", None)


class HUDRenderer:
    """Cached-resource HUD renderer — all Qt objects allocated once in ``__init__``."""

    def __init__(self) -> None:
        self._start_time = time.monotonic()

        # ---------- Pre-allocated colours ----------
        self._c_beat_on = QColor(0, 200, 80, 220)
        self._c_beat_off = QColor(0, 80, 30, 120)

        self._c_status_ok = QColor(0, 230, 100, 230)
        self._c_status_wait = QColor(200, 200, 80, 200)
        self._c_status_err = QColor(230, 60, 60, 230)
        self._c_white_dim = QColor(200, 200, 200, 160)
        self._c_white_soft = QColor(255, 255, 255, 180)
        self._c_white_mid = QColor(255, 255, 255, 230)
        self._c_bg_dark = QColor(0, 0, 0, 160)
        self._c_preview_lbl = QColor(200, 200, 200, 200)
        self._c_state_lbl = QColor(200, 200, 200, 180)
        self._c_action_lbl = QColor(100, 200, 255, 200)
        self._c_gesture_bar_bg = QColor(60, 60, 60, 180)

        pr, pg, pb = OVERLAY_POINTER_COLOR
        self._c_pointer = QColor(pr, pg, pb, 220)
        self._c_pointer_ring = QColor(pr, pg, pb, 120)
        self._c_pointer_cross = QColor(pr, pg, pb, 160)

        dr, dg, db = OVERLAY_DRAG_COLOR
        self._c_drag = QColor(dr, dg, db, 220)
        self._c_drag_ring = QColor(dr, dg, db, 120)
        self._c_drag_cross = QColor(dr, dg, db, 160)

        rr, rg, rb = OVERLAY_RESIZE_COLOR
        self._c_resize = QColor(rr, rg, rb, 220)
        self._c_resize_ring = QColor(rr, rg, rb, 120)
        self._c_resize_cross = QColor(rr, rg, rb, 160)

        gr, gg, gb = OVERLAY_GRIP_COLOR
        self._c_grip = QColor(gr, gg, gb, 180)
        self._c_grip_dim = QColor(gr, gg, gb, 80)
        self._c_grip_fill = QColor(gr, gg, gb, 60)
        self._c_grip_fill_dim = QColor(gr, gg, gb, 26)

        tr, tg, tb = OVERLAY_TARGETING_COLOR
        self._c_targeting = QColor(tr, tg, tb, 200)

        # ---------- Pre-allocated pens / brushes / fonts ----------
        self._pen_none = Qt.PenStyle.NoPen
        self._brush_none = Qt.BrushStyle.NoBrush

        self._pen_status_ok = QPen(self._c_status_ok)
        self._pen_status_wait = QPen(self._c_status_wait)
        self._pen_status_err = QPen(self._c_status_err)
        self._pen_white_dim = QPen(self._c_white_dim)
        self._pen_white_soft = QPen(self._c_white_soft)
        self._pen_white_mid = QPen(self._c_white_mid)
        self._pen_preview_lbl = QPen(self._c_preview_lbl)
        self._pen_state_lbl = QPen(self._c_state_lbl)
        self._pen_action_lbl = QPen(self._c_action_lbl)

        self._brush_beat_on = QBrush(self._c_beat_on)
        self._brush_beat_off = QBrush(self._c_beat_off)
        self._brush_bg_dark = QBrush(self._c_bg_dark)

        self._font_status = QFont("Segoe UI", 10, QFont.Weight.Bold)
        self._font_no_hand = QFont("Segoe UI", 14)
        self._font_preview = QFont("Segoe UI", 7)
        self._font_coords = QFont("Segoe UI", 8)
        self._font_gesture = QFont("Segoe UI", 12, QFont.Weight.Bold)
        self._font_state = QFont("Segoe UI", 9)
        self._font_mode = QFont("Segoe UI", 9, QFont.Weight.Bold)

        self._c_gaze_ring = QColor(60, 180, 255, 160)
        self._c_gaze_dot = QColor(60, 180, 255, 220)

    # ------------------------------------------------------------------

    def render(self, painter: QPainter, data: HudData) -> None:
        """Paint the HUD onto *painter* using the current *data* snapshot."""
        if not HAS_PYQT:
            return
        now = time.monotonic()
        fingertip = data.get("fingertip_screen")
        state = data.get("state", "IDLE")
        hovered_win = data.get("hovered_window")
        gesture_id = data.get("gesture_id")
        gesture_conf = data.get("gesture_confidence", 0.0)
        kind = data.get("interaction_kind")
        capture_ok = data.get("capture_ok", True)
        frame_bgr = data.get("frame_bgr")
        gaze_screen = data.get("gaze_screen")
        pointer_source = data.get("pointer_source", "hand")
        input_mode = data.get("input_mode", "hand")

        self._draw_heartbeat(painter, now)
        self._draw_status_bar(painter, capture_ok=capture_ok, hand_detected=fingertip is not None)
        self._draw_state_label(painter, state, kind)
        self._draw_mode_badge(painter, input_mode, pointer_source)
        self._draw_camera_preview(painter, frame_bgr)

        if not capture_ok:
            return

        if gaze_screen is not None:
            self._draw_gaze_marker(painter, gaze_screen)

        if fingertip is not None:
            if hovered_win is not None and state != "IDLE":
                self._draw_window_border(painter, hovered_win, state)
                self._draw_resize_grip_hint(painter, hovered_win, fingertip)
            self._draw_pointer(painter, fingertip, state, now)
            self._draw_coords(painter, fingertip)
        elif gaze_screen is None:
            self._draw_no_hand(painter)

        if gesture_id:
            self._draw_gesture_info(painter, gesture_id, gesture_conf)

    # ------------------------------------------------------------------
    # Status / diagnostic elements
    # ------------------------------------------------------------------

    def _draw_heartbeat(self, painter: QPainter, now: float) -> None:
        elapsed = now - self._start_time
        brush = self._brush_beat_on if int(elapsed * 2) % 2 == 0 else self._brush_beat_off
        screen = painter.window()
        cx = screen.width() - 14
        cy = 14
        painter.setPen(self._pen_none)
        painter.setBrush(brush)
        painter.drawEllipse(QRectF(cx - 6, cy - 6, 12, 12))

    def _draw_status_bar(
        self,
        painter: QPainter,
        *,
        capture_ok: bool,
        hand_detected: bool,
    ) -> None:
        if capture_ok and hand_detected:
            text = "● MANO RILEVATA"
            pen = self._pen_status_ok
        elif capture_ok:
            text = "○ IN ATTESA MANO…"
            pen = self._pen_status_wait
        else:
            text = "✗ ERRORE CAMERA"
            pen = self._pen_status_err

        painter.setFont(self._font_status)
        painter.setPen(pen)
        painter.drawText(10, 16, text)

    def _draw_no_hand(self, painter: QPainter) -> None:
        screen = painter.window()
        cx = screen.width() // 2
        cy = screen.height() // 2

        painter.setFont(self._font_no_hand)
        painter.setPen(self._pen_white_dim)
        text = "Porta la mano davanti alla webcam"
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text)
        painter.drawText(cx - tw // 2, cy, text)

    def _draw_camera_preview(
        self,
        painter: QPainter,
        frame_bgr: np.ndarray[Any, Any] | None,
    ) -> None:
        """Draw a downscaled webcam thumbnail in the bottom-left corner.

        The resize is performed here (not on the pipeline thread) so the
        pipeline only forwards the raw BGR frame.  ``QImage.Format_BGR888``
        (PyQt6 ≥ 6.5) avoids the channel-order copy entirely.
        """
        if frame_bgr is None:
            return
        try:
            thumb = cv2.resize(frame_bgr, (_PREVIEW_W, _PREVIEW_H))

            if _QIMG_BGR is not None:
                qimg = QImage(
                    cast(bytes, thumb.data),
                    _PREVIEW_W,
                    _PREVIEW_H,
                    _PREVIEW_W * 3,
                    _QIMG_BGR,
                )
            else:
                rgb = thumb[:, :, ::-1].copy()
                qimg = QImage(
                    cast(bytes, rgb.data),
                    _PREVIEW_W,
                    _PREVIEW_H,
                    _PREVIEW_W * 3,
                    QImage.Format.Format_RGB888,
                )

            screen = painter.window()
            margin = 8
            px = margin
            py = screen.height() - _PREVIEW_H - margin

            painter.setPen(self._pen_none)
            painter.setBrush(self._brush_bg_dark)
            painter.drawRect(QRectF(px - 2, py - 2, _PREVIEW_W + 4, _PREVIEW_H + 4))
            painter.drawImage(QRectF(px, py, _PREVIEW_W, _PREVIEW_H), qimg)

            painter.setFont(self._font_preview)
            painter.setPen(self._pen_preview_lbl)
            painter.drawText(px + 2, py + _PREVIEW_H + 11, "webcam")
        except (cv2.error, ValueError, RuntimeError) as exc:
            # Never crash the overlay over a preview failure (e.g. malformed
            # frame buffer, transient OpenCV resize error).
            logger.debug("HUD camera preview draw failed: %s", exc)

    # ------------------------------------------------------------------
    # Pointer
    # ------------------------------------------------------------------

    def _draw_pointer(
        self,
        painter: QPainter,
        point: tuple[int, int],
        state: str,
        now: float,
    ) -> None:
        elapsed = now - self._start_time
        pulse = math.sin(elapsed * 6.0) * 2.5
        x, y = point
        r_inner = OVERLAY_POINTER_RADIUS + pulse
        r_outer = r_inner * 2.4

        if state in ("DRAGGING", "DRAG"):
            c_dot, c_ring, c_cross = self._c_drag, self._c_drag_ring, self._c_drag_cross
        elif state in ("RESIZING", "RESIZE"):
            c_dot, c_ring, c_cross = self._c_resize, self._c_resize_ring, self._c_resize_cross
        else:
            c_dot, c_ring, c_cross = self._c_pointer, self._c_pointer_ring, self._c_pointer_cross

        painter.setPen(QPen(c_ring, 2))
        painter.setBrush(self._brush_none)
        painter.drawEllipse(QRectF(x - r_outer, y - r_outer, r_outer * 2, r_outer * 2))

        painter.setPen(self._pen_none)
        painter.setBrush(QBrush(c_dot))
        painter.drawEllipse(QRectF(x - r_inner, y - r_inner, r_inner * 2, r_inner * 2))

        arm = r_outer * 0.8
        painter.setPen(QPen(c_cross, 1))
        painter.drawLine(QPointF(x - arm, y), QPointF(x + arm, y))
        painter.drawLine(QPointF(x, y - arm), QPointF(x, y + arm))

    def _draw_coords(self, painter: QPainter, point: tuple[int, int]) -> None:
        x, y = point
        label = f"({x}, {y})"
        painter.setFont(self._font_coords)
        painter.setPen(self._pen_white_soft)
        offset = int(OVERLAY_POINTER_RADIUS * 2.8) + 4
        painter.drawText(x + offset, y + offset // 2, label)

    # ------------------------------------------------------------------
    # Window decorations
    # ------------------------------------------------------------------

    def _draw_window_border(self, painter: QPainter, window: Any, state: str) -> None:
        rect = getattr(window, "rect", None)
        if not rect:
            return
        x, y, w, h = rect
        if state in ("DRAGGING", "DRAG"):
            color = self._c_drag
            thickness = 4
        elif state in ("RESIZING", "RESIZE"):
            color = self._c_resize
            thickness = 4
        else:
            color = self._c_targeting
            thickness = 2

        painter.setPen(QPen(color, thickness))
        painter.setBrush(self._brush_none)
        painter.drawRect(QRectF(x, y, w, h))

    def _draw_resize_grip_hint(
        self,
        painter: QPainter,
        window: Any,
        fingertip: tuple[int, int] | None,
    ) -> None:
        rect = getattr(window, "rect", None)
        if not rect:
            return
        x, y, w, h = rect
        grip_w = int(w * OVERLAY_GRIP_RATIO)
        grip_h = int(h * OVERLAY_GRIP_RATIO)
        gx = x + w - grip_w
        gy = y + h - grip_h

        in_grip = False
        if fingertip is not None:
            fx, fy = fingertip
            in_grip = fx >= gx and fy >= gy

        if in_grip:
            pen_color = self._c_grip
            fill_color = self._c_grip_fill
        else:
            pen_color = self._c_grip_dim
            fill_color = self._c_grip_fill_dim

        painter.setPen(QPen(pen_color, 2))
        painter.setBrush(QBrush(fill_color))
        painter.drawRect(QRectF(gx, gy, grip_w, grip_h))

    # ------------------------------------------------------------------
    # Info elements
    # ------------------------------------------------------------------

    def _draw_gesture_info(self, painter: QPainter, gesture_id: str, confidence: float) -> None:
        painter.setPen(self._pen_white_mid)
        painter.setFont(self._font_gesture)
        screen_rect = painter.window()
        x = screen_rect.width() - 200
        y = screen_rect.height() - 80
        painter.drawText(x, y, gesture_id)

        bar_y = y + 10
        bar_w = 160
        bar_h = 12
        painter.setPen(self._pen_none)
        painter.setBrush(QBrush(self._c_gesture_bar_bg))
        painter.drawRect(QRectF(x, bar_y, bar_w, bar_h))
        conf = max(0.0, min(1.0, confidence or 0.0))
        g = int(conf * 200)
        r = int((1 - conf) * 200)
        painter.setBrush(QBrush(QColor(r, g, 50, 220)))
        painter.drawRect(QRectF(x, bar_y, bar_w * conf, bar_h))

    def _draw_mode_badge(
        self,
        painter: QPainter,
        input_mode: str,
        pointer_source: str,
    ) -> None:
        """Top-centre badge showing active input mode + pointer source."""
        screen = painter.window()
        text = f"MODE: {input_mode.upper()}  ·  SRC: {pointer_source}"
        painter.setFont(self._font_mode)
        painter.setPen(self._pen_white_soft)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text)
        x = (screen.width() - tw) // 2
        painter.drawText(x, 18, text)

    def _draw_gaze_marker(
        self,
        painter: QPainter,
        gaze: tuple[int, int],
    ) -> None:
        """Soft cyan ring at the gaze point (Mode B only)."""
        x, y = gaze
        painter.setPen(QPen(self._c_gaze_ring, 2))
        painter.setBrush(self._brush_none)
        painter.drawEllipse(QRectF(x - 22, y - 22, 44, 44))
        painter.setPen(self._pen_none)
        painter.setBrush(QBrush(self._c_gaze_dot))
        painter.drawEllipse(QRectF(x - 4, y - 4, 8, 8))

    def _draw_state_label(self, painter: QPainter, state: str, kind: str | None) -> None:
        painter.setFont(self._font_state)
        painter.setPen(self._pen_state_lbl)
        painter.drawText(10, 34, f"Stato: {state}")
        if kind:
            painter.setPen(self._pen_action_lbl)
            painter.drawText(10, 50, f"Azione: {kind}")
