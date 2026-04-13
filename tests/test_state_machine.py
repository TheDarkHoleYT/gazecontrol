"""Test per IntentStateMachine — verifica tutte le transizioni FSM."""
import pytest

from gazecontrol.intent.state_machine import IntentStateMachine

WINDOW_A = {'hwnd': 1001, 'title': 'Test', 'rect': (100, 100, 400, 300)}
WINDOW_B = {'hwnd': 1002, 'title': 'Other', 'rect': (600, 100, 400, 300)}


def _machine():
    return IntentStateMachine()


def test_initial_state_is_idle():
    m = _machine()
    assert m.state == 'IDLE'


def test_idle_to_targeting_on_window():
    m = _machine()
    m.update(gaze_point=(200, 200), target_window=WINDOW_A,
             gesture_id=None, gesture_confidence=0.0)
    assert m.state == 'TARGETING'


def test_targeting_to_idle_on_window_change():
    m = _machine()
    m.update((200, 200), WINDOW_A, None, 0.0)
    assert m.state == 'TARGETING'
    m.update((200, 200), WINDOW_B, None, 0.0)
    assert m.state == 'IDLE'


def test_targeting_to_ready_after_dwell():
    import time
    m = _machine()
    m.update((200, 200), WINDOW_A, None, 0.0)
    # Simula dwell: modifica _dwell_start manualmente
    m._dwell_start -= cfg.DWELL_TIME_MS / 1000.0 + 0.05
    m.update((200, 200), WINDOW_A, None, 0.0)
    assert m.state == 'READY'


def test_ready_to_acting_on_gesture():
    m = _machine()
    m.state = 'READY'
    m._target_window = WINDOW_A
    m._ready_start = 0.0  # dummy (timeout lontano)
    import time
    m._ready_start = time.time()
    action = m.update((200, 200), WINDOW_A, 'CLOSE_SIGN', 0.95)
    assert m.state == 'COOLDOWN'
    assert action is not None
    assert action['type'] == 'CLOSE'


def test_ready_timeout_returns_to_idle():
    import time
    m = _machine()
    m.state = 'READY'
    m._target_window = WINDOW_A
    m._ready_start = time.time() - cfg.READY_TIMEOUT_S - 1.0
    m.update((200, 200), WINDOW_A, None, 0.0)
    assert m.state == 'IDLE'


def test_drag_lifecycle():
    import time
    m = _machine()
    m.state = 'READY'
    m._target_window = WINDOW_A
    m._ready_start = time.time()
    hand = (200.0, 200.0)

    # Start drag
    action = m.update((200, 200), WINDOW_A, 'GRAB', 0.95, hand_position=hand)
    assert m.state == 'ACTING'
    assert action['data']['phase'] == 'start'

    # Move drag
    hand2 = (250.0, 220.0)
    action = m.update((200, 200), WINDOW_A, 'GRAB', 0.95, hand_position=hand2)
    assert action is not None
    assert action['data']['phase'] == 'move'
    assert action['data']['delta'] == (50.0, 20.0)

    # End drag
    action = m.update((200, 200), WINDOW_A, 'RELEASE', 0.95)
    assert m.state == 'COOLDOWN'
    assert action['data']['phase'] == 'end'


def test_resize_lifecycle():
    import time
    m = _machine()
    m.state = 'READY'
    m._target_window = WINDOW_A
    m._ready_start = time.time()
    hand = (200.0, 200.0)

    action = m.update((200, 200), WINDOW_A, 'PINCH', 0.95, hand_position=hand)
    assert m.state == 'ACTING'
    assert action['data']['phase'] == 'start'

    # Move resize — deve passare delta
    hand2 = (230.0, 250.0)
    action = m.update((200, 200), WINDOW_A, 'PINCH', 0.95, hand_position=hand2)
    assert action is not None
    assert action['data']['phase'] == 'move'
    assert action['data']['delta'] == (30.0, 50.0)


def test_double_pinch_closes_app():
    import time
    m = _machine()
    # Primo pinch
    m.update((200, 200), WINDOW_A, 'PINCH', 0.95)
    m._pinch_active = False
    m._last_pinch_end_time = time.time() - 0.5  # 0.5s fa

    # Secondo pinch entro la finestra
    action = m.update((200, 200), WINDOW_A, 'PINCH', 0.95)
    assert action is not None
    assert action['type'] == 'CLOSE_APP'


def test_cooldown_returns_to_idle():
    import time
    m = _machine()
    m.state = 'COOLDOWN'
    m._cooldown_start = time.time() - cfg.COOLDOWN_MS / 1000.0 - 0.05
    m.update(None, None, None, 0.0)
    assert m.state == 'IDLE'


if __name__ == '__main__':
    test_initial_state_is_idle()
    test_idle_to_targeting_on_window()
    test_targeting_to_idle_on_window_change()
    test_targeting_to_ready_after_dwell()
    test_ready_to_acting_on_gesture()
    test_ready_timeout_returns_to_idle()
    test_drag_lifecycle()
    test_resize_lifecycle()
    test_double_pinch_closes_app()
    test_cooldown_returns_to_idle()
    print("Tutti i test IntentStateMachine superati.")
