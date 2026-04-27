"""GazeMapper — fit/predict/save/load roundtrip with synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

# sklearn required for fit; skip suite when not installed.
pytest.importorskip("sklearn")

from gazecontrol.gaze.gaze_mapper import GazeMapper


def _synthetic_calibration():
    rng = np.random.default_rng(42)
    angles = np.array(
        [(yaw, pitch) for yaw in range(-20, 21, 5) for pitch in range(-15, 16, 5)],
        dtype=float,
    )
    sw, sh = 1920, 1080
    targets = np.column_stack(
        [
            sw / 2 + angles[:, 0] * 30 + rng.normal(0, 5, len(angles)),
            sh / 2 - angles[:, 1] * 25 + rng.normal(0, 5, len(angles)),
        ]
    )
    return angles, targets, sw, sh


def test_unfitted_predict_returns_none():
    mapper = GazeMapper()
    assert mapper.predict(0.0, 0.0) is None


def test_fit_then_predict_within_screen():
    angles, targets, sw, sh = _synthetic_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh)
    err = mapper.fit(angles, targets)
    assert mapper.is_fitted is True
    assert err < 200  # synthetic data → very small error
    px, py = mapper.predict(0.0, 0.0)
    assert 0 <= px < sw and 0 <= py < sh


def test_save_load_roundtrip(tmp_path):
    angles, targets, sw, sh = _synthetic_calibration()
    a = GazeMapper(screen_w=sw, screen_h=sh)
    a.fit(angles, targets)
    a.save(tmp_path / "test_profile")

    b = GazeMapper(screen_w=sw, screen_h=sh)
    assert b.load(tmp_path / "test_profile") is True
    assert b.is_fitted is True
    px_a, py_a = a.predict(5.0, -3.0)
    px_b, py_b = b.predict(5.0, -3.0)
    assert px_a == pytest.approx(px_b)
    assert py_a == pytest.approx(py_b)


def test_load_missing_file_returns_false(tmp_path):
    mapper = GazeMapper()
    assert mapper.load(tmp_path / "no_such") is False


def test_predict_clamps_to_screen():
    angles, targets, sw, sh = _synthetic_calibration()
    mapper = GazeMapper(screen_w=sw, screen_h=sh)
    mapper.fit(angles, targets)
    px, py = mapper.predict(180.0, 90.0)  # extreme angles
    assert 0 <= px <= sw - 1 and 0 <= py <= sh - 1


def test_save_is_atomic_keeps_old_profile_on_failure(tmp_path, monkeypatch):
    """Regression for BUG-004: if the save crashes mid-write the prior
    profile must survive.  Staged ``.part`` files must not leak."""
    angles, targets, sw, sh = _synthetic_calibration()
    a = GazeMapper(screen_w=sw, screen_h=sh)
    a.fit(angles, targets)
    a.save(tmp_path / "profile")

    npz_path = tmp_path / "profile.npz"
    meta_path = tmp_path / "profile.meta.json"
    assert npz_path.exists() and meta_path.exists()
    original_npz = npz_path.read_bytes()
    original_meta = meta_path.read_text(encoding="utf-8")

    # Force np.savez_compressed to raise after a partial write.
    import gazecontrol.gaze.gaze_mapper as gm

    def boom(*args, **kwargs):
        # Touch a .part file to simulate a partial write before crashing.
        target = args[0]
        with open(target, "wb") as fh:
            fh.write(b"corrupt-half-written")
        raise OSError("disk full simulation")

    monkeypatch.setattr(gm.np, "savez_compressed", boom)

    b = GazeMapper(screen_w=sw, screen_h=sh)
    b.fit(angles, targets)
    with pytest.raises(OSError):
        b.save(tmp_path / "profile")

    # Original profile must be intact.
    assert npz_path.read_bytes() == original_npz
    assert meta_path.read_text(encoding="utf-8") == original_meta
    # No leftover .part artefacts.
    leftovers = list(tmp_path.glob("*.part*"))
    assert not leftovers, f"Stale .part files: {leftovers}"
