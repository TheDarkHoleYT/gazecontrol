"""Tests for FingertipMapper — normalized landmark → screen pixel mapping."""

from __future__ import annotations

from gazecontrol.gesture.fingertip_mapper import FingertipMapper, VirtualDesktop


def _desktop(left=0, top=0, width=1920, height=1080) -> VirtualDesktop:
    return VirtualDesktop(left=left, top=top, width=width, height=height)


def _mapper(sensitivity: float = 1.0, **kw) -> FingertipMapper:
    return FingertipMapper(_desktop(**kw), sensitivity=sensitivity)


# ---------------------------------------------------------------------------
# sensitivity=1.0 — original 1:1 mapping
# ---------------------------------------------------------------------------


def test_map_origin_s1():
    assert _mapper().map(0.0, 0.0) == (0, 0)


def test_map_center_s1():
    sx, sy = _mapper().map(0.5, 0.5)
    assert sx == 960
    assert sy == 540


def test_map_far_corner_s1():
    sx, sy = _mapper().map(1.0, 1.0)
    assert sx == 1920
    assert sy == 1080


def test_map_top_right_s1():
    m = FingertipMapper(_desktop(width=2560, height=1440), sensitivity=1.0)
    assert m.map(1.0, 0.0) == (2560, 0)


def test_map_with_virtual_offset_s1():
    m = FingertipMapper(_desktop(left=-1920, top=0, width=3840, height=1080), sensitivity=1.0)
    sx, sy = m.map(0.0, 0.5)
    assert sx == -1920
    assert sy == 540


# ---------------------------------------------------------------------------
# sensitivity=2.0 — central 50 % of frame → full screen
# ---------------------------------------------------------------------------


def test_center_maps_to_center_s2():
    """Centre of frame always maps to centre of screen regardless of sensitivity."""
    sx, sy = _mapper(2.0).map(0.5, 0.5)
    assert sx == 960
    assert sy == 540


def test_quarter_maps_to_screen_left_s2():
    """lm_x=0.25 with s=2 → nx=(0.25-0.5)*2+0.5=0.0 → screen left edge."""
    sx, _ = _mapper(2.0).map(0.25, 0.5)
    assert sx == 0


def test_three_quarter_maps_to_screen_right_s2():
    """lm_x=0.75 with s=2 → nx=(0.75-0.5)*2+0.5=1.0 → screen right edge."""
    sx, _ = _mapper(2.0).map(0.75, 0.5)
    assert sx == 1920


def test_corners_clamped_s2():
    """Landmarks at frame edges are clamped to screen edges with s=2."""
    m = _mapper(2.0)
    assert m.map(0.0, 0.0) == (0, 0)
    assert m.map(1.0, 1.0) == (1920, 1080)


def test_sensitivity_property():
    m = _mapper(3.0)
    assert m.sensitivity == 3.0


# ---------------------------------------------------------------------------
# sensitivity clamp — minimum enforced
# ---------------------------------------------------------------------------


def test_sensitivity_minimum_enforced():
    """Sensitivity < 0.5 is clamped to 0.5 (never zero or negative)."""
    m = FingertipMapper(_desktop(), sensitivity=0.0)
    assert m.sensitivity >= 0.1


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_desktop_property():
    d = _desktop(left=10, top=20, width=1920, height=1080)
    m = FingertipMapper(d)
    assert m.desktop is d


def test_virtual_desktop_from_win32_fallback():
    """from_win32 falls back to 1920×1080 when Win32 is unavailable."""
    import unittest.mock as mock

    with mock.patch("ctypes.windll", side_effect=AttributeError):
        d = VirtualDesktop.from_win32()
    assert d.width > 0
    assert d.height > 0
