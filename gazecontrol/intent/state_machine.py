import time

from gazecontrol.config import (
    DWELL_TIME_MS, READY_TIMEOUT_S, COOLDOWN_MS,
    GESTURE_CONFIDENCE_THRESHOLD,
)

GESTURE_ACTION_MAP = {
    'GRAB': 'DRAG',
    'RELEASE': 'RELEASE',
    'PINCH': 'RESIZE',
    'SWIPE_LEFT': 'MINIMIZE',
    'SWIPE_RIGHT': 'BRING_FRONT',
    'CLOSE_SIGN': 'CLOSE',
    'SCROLL_UP': 'SCROLL_UP',
    'SCROLL_DOWN': 'SCROLL_DOWN',
    'MAXIMIZE': 'MAXIMIZE',
}

CONTINUOUS_GESTURES = {'GRAB', 'PINCH'}

# Finestra temporale per riconoscere il doppio pinch (secondi)
DOUBLE_PINCH_WINDOW_S = 1.5


class IntentStateMachine:
    def __init__(self):
        self.state = 'IDLE'
        self._target_window = None
        self._dwell_start = None
        self._ready_start = None
        self._cooldown_start = None
        self._drag_start_hand = None    # posizione mano all'inizio del drag (px schermo)
        self._drag_start_rect = None
        self._resize_start_hand = None  # posizione mano all'inizio del resize
        self._resize_start_rect = None
        self._active_gesture = None

        # Rilevamento doppio pinch
        self._pinch_active = False
        self._last_pinch_end_time = None

    def update(self, gaze_point, target_window, gesture_id, gesture_confidence,
               hand_position=None):
        """
        Args:
            gaze_point: (x, y) pixel schermo oppure None.
            target_window: finestra sotto il punto di sguardo oppure None.
            gesture_id: stringa gesture classificata oppure None.
            gesture_confidence: float [0,1].
            hand_position: (x, y) posizione polso in pixel schermo, oppure None.
                           Usato per il calcolo del delta durante il drag.
        """
        now = time.time()

        # ------------------------------------------------------------------
        # Rilevamento doppio pinch — funziona in QUALSIASI stato FSM.
        # Due PINCH separati entro DOUBLE_PINCH_WINDOW_S → CLOSE_APP.
        # ------------------------------------------------------------------
        if gesture_id == 'PINCH' and gesture_confidence > GESTURE_CONFIDENCE_THRESHOLD:
            if not self._pinch_active:
                self._pinch_active = True
                if (self._last_pinch_end_time is not None and
                        now - self._last_pinch_end_time < DOUBLE_PINCH_WINDOW_S):
                    # Secondo pinch rilevato: chiudi l'applicazione
                    self._pinch_active = False
                    self._last_pinch_end_time = None
                    return {'type': 'CLOSE_APP'}
        elif self._pinch_active and gesture_id != 'PINCH':
            self._pinch_active = False
            self._last_pinch_end_time = now

        # ------------------------------------------------------------------
        # FSM principale
        # ------------------------------------------------------------------
        if self.state == 'IDLE':
            if target_window and gaze_point:
                self._target_window = target_window
                self._dwell_start = now
                self.state = 'TARGETING'
            return None

        if self.state == 'TARGETING':
            if not target_window or (self._target_window and
                    target_window.get('hwnd') != self._target_window.get('hwnd')):
                self.state = 'IDLE'
                self._target_window = None
                self._dwell_start = None
                return None
            elapsed_ms = (now - self._dwell_start) * 1000
            if elapsed_ms >= DWELL_TIME_MS:
                self.state = 'READY'
                self._ready_start = now
            return None

        if self.state == 'READY':
            if now - self._ready_start > READY_TIMEOUT_S:
                self.state = 'IDLE'
                self._target_window = None
                self._ready_start = None
                return None
            if gesture_id and gesture_confidence and gesture_confidence > GESTURE_CONFIDENCE_THRESHOLD:
                action_type = GESTURE_ACTION_MAP.get(gesture_id)
                if not action_type or action_type == 'RELEASE':
                    return None
                self.state = 'ACTING'
                self._active_gesture = gesture_id
                if gesture_id == 'GRAB':
                    # Usa la posizione della mano (non lo sguardo) come punto di partenza
                    self._drag_start_hand = hand_position
                    self._drag_start_rect = self._target_window.get('rect')
                    return {
                        'type': 'DRAG',
                        'window': self._target_window,
                        'data': {'phase': 'start'},
                    }
                if gesture_id == 'PINCH':
                    self._resize_start_rect = self._target_window.get('rect')
                    self._resize_start_hand = hand_position
                    return {
                        'type': 'RESIZE',
                        'window': self._target_window,
                        'data': {'phase': 'start'},
                    }
                action = {
                    'type': action_type,
                    'window': self._target_window,
                    'data': {},
                }
                self.state = 'COOLDOWN'
                self._cooldown_start = now
                self._active_gesture = None
                return action
            return None

        if self.state == 'ACTING':
            if self._active_gesture == 'GRAB':
                if gesture_id == 'RELEASE' or gesture_id != 'GRAB':
                    self.state = 'COOLDOWN'
                    self._cooldown_start = now
                    self._active_gesture = None
                    self._drag_start_hand = None
                    self._drag_start_rect = None
                    return {
                        'type': 'DRAG',
                        'window': self._target_window,
                        'data': {'phase': 'end'},
                    }
                # Aggiorna posizione finestra in base al movimento della mano
                if hand_position and self._drag_start_hand:
                    dx = hand_position[0] - self._drag_start_hand[0]
                    dy = hand_position[1] - self._drag_start_hand[1]
                    return {
                        'type': 'DRAG',
                        'window': self._target_window,
                        'data': {
                            'phase': 'move',
                            'delta': (dx, dy),
                            'start_rect': self._drag_start_rect,
                        },
                    }
                return None

            if self._active_gesture == 'PINCH':
                if gesture_id != 'PINCH':
                    self.state = 'COOLDOWN'
                    self._cooldown_start = now
                    self._active_gesture = None
                    self._resize_start_rect = None
                    self._resize_start_hand = None
                    return {
                        'type': 'RESIZE',
                        'window': self._target_window,
                        'data': {'phase': 'end'},
                    }
                if hand_position and self._resize_start_hand:
                    dx = hand_position[0] - self._resize_start_hand[0]
                    dy = hand_position[1] - self._resize_start_hand[1]
                    return {
                        'type': 'RESIZE',
                        'window': self._target_window,
                        'data': {
                            'phase': 'move',
                            'delta': (dx, dy),
                            'start_rect': self._resize_start_rect,
                        },
                    }
                return None

            self.state = 'COOLDOWN'
            self._cooldown_start = now
            self._active_gesture = None
            return None

        if self.state == 'COOLDOWN':
            elapsed_ms = (now - self._cooldown_start) * 1000
            if elapsed_ms >= COOLDOWN_MS:
                self.state = 'IDLE'
                self._target_window = None
                self._cooldown_start = None
            return None

        return None
