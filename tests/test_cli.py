"""Tests for CLI import behaviour — no env side-effects at import time."""

from __future__ import annotations

import importlib
import os


def test_cli_import_has_no_env_side_effect():
    """Importing gazecontrol.cli must not mutate os.environ."""
    before = dict(os.environ)
    import gazecontrol.cli

    importlib.reload(gazecontrol.cli)
    # Only keys that existed before the import should be present, unchanged.
    assert os.environ == before, (
        f"Import of gazecontrol.cli mutated os.environ: "
        f"diff={set(os.environ.items()) ^ set(before.items())}"
    )
