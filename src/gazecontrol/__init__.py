"""GazeControl — hand gesture + eye tracking desktop control.

Stable public API (PEP 561, ``py.typed`` shipped):

    >>> from gazecontrol import __version__, AppSettings, get_settings, InputMode

Anything not listed in :data:`__all__` is internal and may change without
notice between minor releases — see ``docs/adr/0006-public-api-contract.md``.
"""

__version__ = "0.8.0"

from gazecontrol.runtime.input_mode import InputMode
from gazecontrol.settings import AppSettings, get_settings

__all__ = [
    "AppSettings",
    "InputMode",
    "__version__",
    "get_settings",
]
