"""
One Euro Filter — filtro adattivo per eye tracking.

Algoritmo: Casiez et al. (2012) "1€ Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Interactive Systems".

In sintesi:
  - Durante le FIXATION (bassa velocità): bassa cutoff → molto smooth, poco jitter.
  - Durante le SACCADI (alta velocità): alta cutoff → molto reattivo, segue subito.

Parametri chiave:
  min_cutoff : frequenza minima di taglio (Hz). Valori bassi → più smooth a riposo.
  beta       : coefficiente di velocità. Valori alti → più reattivo durante il movimento.
  d_cutoff   : cutoff per il filtro sulla derivata (non richiede tuning).
"""
import math
import time


class _LowPassFilter:
    def __init__(self, cutoff: float, freq: float):
        self._freq = freq
        self._cutoff = cutoff
        self._x = None
        self._dx = 0.0

    def alpha(self, cutoff: float) -> float:
        te = 1.0 / self._freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def step(self, x: float, cutoff: float) -> float:
        a = self.alpha(cutoff)
        if self._x is None:
            self._x = x
        self._x = a * x + (1.0 - a) * self._x
        return self._x

    @property
    def last(self) -> float:
        return self._x if self._x is not None else 0.0


class OneEuroFilter:
    """
    Filtra un segnale scalare con il 1€ Filter.

    Uso tipico (un'istanza per asse x, un'altra per asse y):

        filter_x = OneEuroFilter(freq=30.0, min_cutoff=1.5, beta=0.007)
        filter_y = OneEuroFilter(freq=30.0, min_cutoff=1.5, beta=0.007)

        for t, (raw_x, raw_y) in data:
            sx = filter_x.filter(raw_x, timestamp=t)
            sy = filter_y.filter(raw_y, timestamp=t)
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.5,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self._freq = freq
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filter = _LowPassFilter(min_cutoff, freq)
        self._dx_filter = _LowPassFilter(d_cutoff, freq)
        self._last_ts: float | None = None

    def filter(self, x: float, timestamp: float | None = None) -> float:
        """
        Applica il filtro al valore x.

        Args:
            x         : valore raw da filtrare.
            timestamp : timestamp in secondi (default: time.time()).

        Returns:
            Valore filtrato.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        if self._last_ts is not None:
            dt = timestamp - self._last_ts
            if dt > 0.0:
                self._freq = 1.0 / dt
                self._x_filter._freq = self._freq
                self._dx_filter._freq = self._freq

        self._last_ts = timestamp

        # Stima derivata
        prev_x = self._x_filter.last
        dx = (x - prev_x) * self._freq

        # Filtra derivata
        edx = self._dx_filter.step(dx, self._d_cutoff)

        # Cutoff adattiva: aumenta con la velocità
        cutoff = self._min_cutoff + self._beta * abs(edx)

        return self._x_filter.step(x, cutoff)

    def reset(self):
        """Resetta lo stato interno (utile dopo un blink lungo o gap di dati)."""
        self._x_filter._x = None
        self._dx_filter._x = None
        self._last_ts = None
