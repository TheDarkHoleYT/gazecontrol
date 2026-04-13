"""Test per OneEuroFilter."""
import math

from gazecontrol.gaze.one_euro_filter import OneEuroFilter


def test_constant_signal_converges():
    """Un segnale costante deve convergere al valore vero."""
    f = OneEuroFilter(freq=30.0, min_cutoff=1.5, beta=0.007)
    val = 500.0
    result = None
    for i in range(100):
        result = f.filter(val, timestamp=i / 30.0)
    assert abs(result - val) < 1.0, f"Atteso {val}, ottenuto {result:.2f}"


def test_tracks_slow_ramp():
    """Il filtro deve seguire un segnale lento con basso errore."""
    f = OneEuroFilter(freq=30.0, min_cutoff=1.5, beta=0.007)
    errors = []
    for i in range(60):
        t = i / 30.0
        x = 100.0 + i * 2.0  # rampa lenta
        out = f.filter(x, timestamp=t)
        if i > 10:  # ignora transitorio iniziale
            errors.append(abs(out - x))
    assert max(errors) < 30.0, f"Errore massimo troppo alto: {max(errors):.1f}"


def test_reset_clears_state():
    """Dopo reset(), il filtro non deve ricordare lo stato precedente."""
    f = OneEuroFilter(freq=30.0)
    for i in range(30):
        f.filter(1000.0, timestamp=i / 30.0)
    f.reset()
    out = f.filter(0.0, timestamp=0.0)
    assert out == 0.0, f"Dopo reset atteso 0.0, ottenuto {out}"


def test_two_independent_axes():
    """Due filtri separati per x e y non si influenzano."""
    fx = OneEuroFilter(freq=30.0, min_cutoff=1.5, beta=0.007)
    fy = OneEuroFilter(freq=30.0, min_cutoff=1.5, beta=0.007)

    for i in range(50):
        t = i / 30.0
        ox = fx.filter(100.0, timestamp=t)
        oy = fy.filter(500.0, timestamp=t)

    assert abs(ox - 100.0) < 2.0
    assert abs(oy - 500.0) < 2.0


def test_saccade_responsiveness():
    """Durante un salto brusco (saccade), il filtro deve seguire rapidamente."""
    f = OneEuroFilter(freq=30.0, min_cutoff=1.5, beta=0.5)  # beta alto = reattivo
    # Prima al 0
    for i in range(30):
        f.filter(0.0, timestamp=i / 30.0)
    # Salto a 500
    out = None
    for i in range(10):
        t = (30 + i) / 30.0
        out = f.filter(500.0, timestamp=t)
    # Deve aver raggiunto almeno l'80% del valore target in 10 frame
    assert out > 400.0, f"Dopo saccade atteso >400, ottenuto {out:.1f}"


if __name__ == '__main__':
    test_constant_signal_converges()
    test_tracks_slow_ramp()
    test_reset_clears_state()
    test_two_independent_axes()
    test_saccade_responsiveness()
    print("Tutti i test OneEuroFilter superati.")
