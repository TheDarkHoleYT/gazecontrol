from .base import BaseWindowManager

try:
    import win32gui
    import win32con
    import win32api
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


class WindowsManager(BaseWindowManager):
    def move_window(self, hwnd, x, y):
        if not HAS_WIN32:
            return
        try:
            rect = win32gui.GetWindowRect(hwnd)
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            win32gui.MoveWindow(hwnd, x, y, w, h, True)
        except Exception:
            pass

    def resize_window(self, hwnd, x, y, w, h):
        if not HAS_WIN32:
            return
        try:
            win32gui.MoveWindow(hwnd, x, y, w, h, True)
        except Exception:
            pass

    def close_window(self, hwnd):
        if not HAS_WIN32:
            return
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        except Exception:
            pass

    def minimize_window(self, hwnd):
        if not HAS_WIN32:
            return
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        except Exception:
            pass

    def maximize_window(self, hwnd):
        if not HAS_WIN32:
            return
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        except Exception:
            pass

    def bring_to_front(self, hwnd):
        if not HAS_WIN32:
            return
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def execute(self, action):
        import ctypes
        action_type = action.get('type')
        if action_type in ('SCROLL_UP', 'SCROLL_DOWN') and HAS_WIN32:
            hwnd = action.get('window', {}).get('hwnd')
            if hwnd:
                try:
                    delta = 120 if action_type == 'SCROLL_UP' else -120
                    # BUG-7 fix: Python ints are unbounded; -120 << 16 produces a large
                    # negative value. Encode delta as unsigned 16-bit high word of WPARAM.
                    high_word = ctypes.c_int16(delta).value & 0xFFFF
                    wparam = high_word << 16
                    win32api.PostMessage(
                        hwnd, win32con.WM_MOUSEWHEEL, wparam, 0
                    )
                except Exception:
                    pass
            return
        super().execute(action)
