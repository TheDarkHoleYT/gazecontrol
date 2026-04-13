from __future__ import annotations

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseWindowManager(ABC):
    """Abstract base for platform-specific window managers.

    Subclasses implement the primitive window operations (move, resize, …).
    Action routing is handled by :meth:`execute`, which acts as a dispatcher.
    Subclasses that need to handle additional action types (e.g. scroll) should
    override ``execute``, handle their specific types, then call ``super().execute``
    for all other types — keeping the dispatch chain intact.
    """

    @abstractmethod
    def move_window(self, hwnd, x: int, y: int) -> None:
        pass

    @abstractmethod
    def resize_window(self, hwnd, x: int, y: int, w: int, h: int) -> None:
        pass

    @abstractmethod
    def close_window(self, hwnd) -> None:
        pass

    @abstractmethod
    def minimize_window(self, hwnd) -> None:
        pass

    @abstractmethod
    def maximize_window(self, hwnd) -> None:
        pass

    @abstractmethod
    def bring_to_front(self, hwnd) -> None:
        pass

    def execute(self, action: dict) -> None:
        """Dispatch an action dict to the appropriate primitive method.

        Action dict shape::

            {
                'type': str,           # e.g. 'DRAG', 'RESIZE', 'CLOSE', …
                'window': {'hwnd': int, ...},
                'data': {...},         # type-specific payload
            }

        Subclasses that handle extra action types (e.g. 'SCROLL_UP') should
        override this method, handle their types, and call ``super().execute(action)``
        for unrecognised types.
        """
        action_type = action.get('type')
        hwnd = action.get('window', {}).get('hwnd')
        data = action.get('data', {})

        if not hwnd:
            return

        if action_type == 'DRAG':
            phase = data.get('phase')
            if phase == 'move':
                start_rect = data.get('start_rect')
                delta = data.get('delta', (0, 0))
                if start_rect:
                    new_x = start_rect[0] + int(delta[0])
                    new_y = start_rect[1] + int(delta[1])
                    self.move_window(hwnd, new_x, new_y)
        elif action_type == 'RESIZE':
            phase = data.get('phase')
            if phase == 'move':
                start_rect = data.get('start_rect')
                delta = data.get('delta', (0, 0))
                if start_rect:
                    # delta x → change width; delta y → change height.
                    # Window stays anchored at its top-left corner.
                    new_w = max(100, start_rect[2] + int(delta[0]))
                    new_h = max(60,  start_rect[3] + int(delta[1]))
                    self.resize_window(hwnd, start_rect[0], start_rect[1],
                                       new_w, new_h)
        elif action_type == 'CLOSE':
            self.close_window(hwnd)
        elif action_type == 'MINIMIZE':
            self.minimize_window(hwnd)
        elif action_type == 'MAXIMIZE':
            self.maximize_window(hwnd)
        elif action_type == 'BRING_FRONT':
            self.bring_to_front(hwnd)
        elif action_type in ('SCROLL_UP', 'SCROLL_DOWN'):
            # Handled by subclasses; silently ignored here.
            pass
        else:
            logger.debug("BaseWindowManager.execute: unhandled action type %r", action_type)
