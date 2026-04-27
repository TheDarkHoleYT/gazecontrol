"""LauncherPanel — semi-transparent overlay panel for launching applications.

Displayed when the user performs a double-pinch.  Each app tile can be
activated by a single-pinch (tap) at its screen position.

Thread safety: all mutations must happen in the Qt main thread.
Visibility is toggled from the pipeline thread via ``OverlayWindow``'s
``data_changed`` signal (``launcher_visible`` field in the HUD dict).
The panel widget renders itself as a child of the overlay window if Qt is
available, or silently degrades otherwise.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen

if TYPE_CHECKING:
    from gazecontrol.window_manager.launcher import AppLauncher, LauncherApp

logger = logging.getLogger(__name__)


class LauncherPanel:
    """Renders an app-launcher grid and provides hit-testing for tap activation.

    Args:
        apps:     Ordered list of :class:`~gazecontrol.window_manager.launcher.LauncherApp`.
        launcher: :class:`~gazecontrol.window_manager.launcher.AppLauncher` instance.
        columns:  Number of columns in the tile grid.
        tile_w:   Tile width in pixels.
        tile_h:   Tile height in pixels.
    """

    _TILE_W = 120
    _TILE_H = 80
    _MARGIN = 16
    _PAD = 8

    def __init__(
        self,
        apps: list[LauncherApp],
        launcher: AppLauncher,
        columns: int = 4,
    ) -> None:
        self._apps = apps
        self._launcher = launcher
        self._columns = max(1, columns)
        self._visible = False

    @property
    def visible(self) -> bool:
        """Whether the panel is currently shown."""
        return self._visible

    def toggle(self) -> None:
        """Show or hide the launcher panel."""
        self._visible = not self._visible
        logger.debug("LauncherPanel: visible=%s", self._visible)

    def show(self) -> None:
        """Make the launcher panel visible."""
        self._visible = True

    def hide(self) -> None:
        """Hide the launcher panel."""
        self._visible = False

    # ------------------------------------------------------------------
    # Rendering (called from HUDRenderer.render() when launcher_visible)
    # ------------------------------------------------------------------

    def render(self, painter: QPainter, screen_w: int, screen_h: int) -> None:
        """Draw the launcher grid onto *painter*."""
        if not self._visible or not self._apps:
            return

        tiles = self._compute_tiles(screen_w, screen_h)
        panel_rect = self._panel_rect(tiles)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(20, 20, 30, 210)))
        painter.drawRoundedRect(QRectF(*panel_rect), 12, 12)

        for (tx, ty, tw, th), app in tiles:
            painter.setBrush(QBrush(QColor(50, 55, 70, 220)))
            painter.setPen(QPen(QColor(80, 90, 110, 200), 1))
            painter.drawRoundedRect(QRectF(tx, ty, tw, th), 6, 6)

            painter.setPen(QPen(QColor(230, 230, 240, 240)))
            font = QFont("Segoe UI", 9)
            painter.setFont(font)
            painter.drawText(
                int(tx + self._PAD),
                int(ty + th // 2 + 5),
                app.name[:18],
            )

    def hit_test(
        self,
        point: tuple[int, int],
        screen_w: int,
        screen_h: int,
    ) -> LauncherApp | None:
        """Return the :class:`LauncherApp` at *point*, or ``None``."""
        if not self._visible or not self._apps:
            return None
        tiles = self._compute_tiles(screen_w, screen_h)
        px, py = point
        for (tx, ty, tw, th), app in tiles:
            if tx <= px <= tx + tw and ty <= py <= ty + th:
                return app
        return None

    def handle_tap(self, point: tuple[int, int], screen_w: int, screen_h: int) -> bool:
        """If *point* hits a tile, launch the app and hide the panel.

        Returns:
            ``True`` if a tile was activated, ``False`` otherwise.
        """
        app = self.hit_test(point, screen_w, screen_h)
        if app is None:
            return False
        self._launcher.launch(app)
        self.hide()
        return True

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _compute_tiles(
        self, screen_w: int, screen_h: int
    ) -> list[tuple[tuple[int, int, int, int], LauncherApp]]:
        """Return list of ``((x, y, w, h), app)`` tuples for each tile."""
        tw = self._TILE_W
        th = self._TILE_H
        pad = self._PAD
        cols = self._columns
        rows = max(1, (len(self._apps) + cols - 1) // cols)

        total_w = cols * tw + (cols - 1) * pad + 2 * self._MARGIN
        total_h = rows * th + (rows - 1) * pad + 2 * self._MARGIN

        ox = (screen_w - total_w) // 2
        oy = (screen_h - total_h) // 2

        tiles = []
        for i, app in enumerate(self._apps):
            col = i % cols
            row = i // cols
            x = ox + self._MARGIN + col * (tw + pad)
            y = oy + self._MARGIN + row * (th + pad)
            tiles.append(((x, y, tw, th), app))
        return tiles

    def _panel_rect(
        self, tiles: list[tuple[tuple[int, int, int, int], LauncherApp]]
    ) -> tuple[int, int, int, int]:
        if not tiles:
            return (0, 0, 0, 0)
        xs = [t[0][0] for t in tiles]
        ys = [t[0][1] for t in tiles]
        xe = [t[0][0] + t[0][2] for t in tiles]
        ye = [t[0][1] + t[0][3] for t in tiles]
        x0 = min(xs) - self._MARGIN
        y0 = min(ys) - self._MARGIN
        x1 = max(xe) + self._MARGIN
        y1 = max(ye) + self._MARGIN
        return (x0, y0, x1 - x0, y1 - y0)
