"""HudState — typed snapshot of HUD display data for hand-only control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HudState:
    """Data required to render one HUD frame.

    Immutable snapshot — callers build a fresh ``HudState`` per tick and
    swap the reference atomically (Python's GIL makes single-attribute
    pointer swaps thread-safe). Frozen prevents in-place mutation that
    would cause cross-thread races between the pipeline producer and the
    Qt renderer.
    """

    fingertip_screen: tuple[int, int] | None = None
    state: str = "IDLE"
    hovered_window: Any = None  # HoveredWindow | None
    gesture_id: str | None = None
    gesture_confidence: float = 0.0
    interaction_kind: str | None = None  # InteractionKind value or None
    launcher_visible: bool = False
    # Eye-tracking enrichment (None / 0 when input_mode == HAND_ONLY).
    gaze_screen: tuple[int, int] | None = None
    gaze_confidence: float = 0.0
    pointer_source: str = "hand"
    input_mode: str = "hand"
    # --- v1.0 G20 — HUD quality feedback fields. ---
    #: True while DriftCorrector is collecting an explicit-recenter
    #: dwell (G7). The renderer should pulse the gaze ring instead of
    #: drawing the regular pointer.
    recenter_active: bool = False
    #: Magnitude in pixels of the active drift offset (DriftCorrector
    #: telemetry). Drawn as a small label next to the gaze marker so
    #: operators see when the correction is creeping toward the cap.
    gaze_drift_px: float = 0.0
    #: True when the convergence test (G7) reports the offset has
    #: stabilised — the HUD shows a "stable" hint.
    gaze_converged: bool = False
    #: True after FusionSettings.gaze_failure_threshold_frames
    #: consecutive prediction failures (ADR-0008 / G17). The HUD
    #: surfaces a degraded badge so the operator knows the pointer
    #: is hand-only until recovery.
    gaze_backend_down: bool = False
    #: Per-frame drop ratio over the recent profiler window. Red
    #: numbers above 5 % are the HUD's signal to investigate CPU /
    #: camera health.
    dropped_frame_ratio: float = 0.0
    #: Last per-prediction gaze error vs. ground truth (replay mode,
    #: G12). NaN when no replay source is active so the renderer can
    #: omit the field entirely instead of drawing "0 px".
    gaze_error_px: float = float("nan")
