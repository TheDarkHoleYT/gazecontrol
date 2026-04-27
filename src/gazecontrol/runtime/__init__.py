"""Runtime selection — input mode, mode persistence, pipeline factory."""

from gazecontrol.runtime.input_mode import (
    InputMode,
    load_persisted_mode,
    persist_mode,
)

__all__ = ["InputMode", "load_persisted_mode", "persist_mode"]
