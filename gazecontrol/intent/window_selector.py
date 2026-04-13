import time

try:
    import win32gui
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


class WindowSelector:
    def __init__(self, cache_interval=0.5):
        self._cache_interval = cache_interval
        self._cache = []
        self._last_update = 0

    def _refresh_cache(self):
        now = time.time()
        if now - self._last_update < self._cache_interval and self._cache:
            return
        self._last_update = now
        if not HAS_WIN32:
            self._cache = []
            return
        windows = []

        def enum_callback(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return
            try:
                rect = win32gui.GetWindowRect(hwnd)
            except Exception:
                return
            x, y, x2, y2 = rect
            w = x2 - x
            h = y2 - y
            if w <= 0 or h <= 0:
                return
            windows.append({
                'hwnd': hwnd,
                'title': title,
                'rect': (x, y, w, h),
            })

        try:
            win32gui.EnumWindows(enum_callback, None)
        except Exception:
            pass
        self._cache = windows

    def find_window(self, gaze_point):
        if not gaze_point:
            return None
        self._refresh_cache()
        gx, gy = gaze_point[0], gaze_point[1]
        for win in self._cache:
            x, y, w, h = win['rect']
            if x <= gx <= x + w and y <= gy <= y + h:
                return win
        return None
