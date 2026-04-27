"""FingertipMapper — maps normalized MediaPipe landmark coords to screen pixels.

The frame grabbed by ``FrameGrabber`` is already horizontally flipped
(``cv2.flip(frame, 1)``), so landmark x=0 corresponds to the left edge of
the *mirrored* frame.  No additional mirroring is required here.

Pointer filter pipeline
-----------------------
Raw landmark coordinates pass through four optional stages before being
scaled to screen pixels:

1. **One-Euro filter** (per-axis) — removes landmark jitter at rest while
   staying responsive during fast movement.
2. **Kalman filter** (2-D) — constant-velocity prediction compensates for
   pipeline latency and provides temporal continuity.
3. **Dead-zone** — suppresses micro-jitter when the hand is stationary;
   releases smoothly once movement exceeds the hysteresis threshold.
4. **Acceleration curve** — non-linear velocity-to-gain mapping so slow
   precise movements are dampened and fast navigation sweeps are amplified.

All filters can be individually enabled/disabled via
``GestureSettings.pointer.*``.

Sensitivity
-----------
After filtering the normalised coordinates are scaled around the frame centre
by ``sensitivity``::

    nx = clamp((lm_x - 0.5) * sensitivity + 0.5,  0.0, 1.0)
    ny = clamp((lm_y - 0.5) * sensitivity + 0.5,  0.0, 1.0)
    screen_x = desktop.left + nx * desktop.width
    screen_y = desktop.top  + ny * desktop.height

``sensitivity = 2.0``  → central 50 % of frame → full screen (default)
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gazecontrol.filters.acceleration_curve import AccelerationCurve
from gazecontrol.filters.dead_zone import DeadZone
from gazecontrol.filters.kalman import KalmanFilter2D
from gazecontrol.filters.one_euro import OneEuroFilter

if TYPE_CHECKING:
    from gazecontrol.settings import PointerFilterSettings


@dataclass(frozen=True)
class VirtualDesktop:
    """Bounding box of the Windows virtual desktop (all monitors combined).

    Attributes:
        left:   X origin of the leftmost monitor (may be negative).
        top:    Y origin of the topmost monitor (may be negative).
        width:  Total pixel width across all monitors.
        height: Total pixel height across all monitors.
    """

    left: int
    top: int
    width: int
    height: int

    @classmethod
    def from_win32(cls) -> VirtualDesktop:
        """Query the current virtual-desktop geometry via Win32 GetSystemMetrics.

        Falls back to a sensible 1920×1080 primary-only desktop if the Win32
        call fails (e.g. when running in a test environment without a display).
        """
        try:
            user32 = ctypes.windll.user32
            SM_XVIRTUALSCREEN = 76
            SM_YVIRTUALSCREEN = 77
            SM_CXVIRTUALSCREEN = 78
            SM_CYVIRTUALSCREEN = 79
            left = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
            top = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
            width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
            height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
            if width > 0 and height > 0:
                return cls(left=left, top=top, width=width, height=height)
        except Exception:
            pass
        return cls(left=0, top=0, width=1920, height=1080)


class FingertipMapper:
    """Map a normalized MediaPipe landmark to virtual-desktop pixel coordinates.

    Applies a configurable filter pipeline (One-Euro → Kalman → dead-zone →
    acceleration curve) before sensitivity scaling and screen projection.

    Args:
        desktop:      Virtual-desktop bounding box.
        sensitivity:  Movement amplification factor (> 1 = less hand movement
                      needed).  ``2.0`` means moving your hand in the central
                      50 % of the camera frame covers the full screen.
        filter_cfg:   Optional pointer filter configuration.  When *None*,
                      default filter parameters are used.  Pass
                      ``settings.gesture.pointer`` to use app settings.

    Usage::

        desktop = VirtualDesktop.from_win32()
        mapper  = FingertipMapper(desktop, sensitivity=2.0)
        sx, sy  = mapper.map(lm_x, lm_y, dt=0.033, confidence=0.95)
    """

    def __init__(
        self,
        desktop: VirtualDesktop,
        sensitivity: float = 2.0,
        filter_cfg: PointerFilterSettings | None = None,
    ) -> None:
        self._desktop = desktop
        self._sensitivity = max(0.1, float(sensitivity))

        # Build filter instances from config (or use defaults).
        self._oe_x: OneEuroFilter | None = None
        self._oe_y: OneEuroFilter | None = None
        self._kalman: KalmanFilter2D | None = None
        self._dead_zone: DeadZone | None = None
        self._accel: AccelerationCurve | None = None

        self._build_filters(filter_cfg)

        # Previous normalised position — used for velocity estimation.
        self._prev_nx: float | None = None
        self._prev_ny: float | None = None
        self._prev_ts: float = time.monotonic()

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def _build_filters(self, cfg: PointerFilterSettings | None) -> None:
        """Instantiate filters from *cfg*.

        When *cfg* is *None* all filters are disabled — the mapper behaves
        identically to the original single-line sensitivity scaling.  Pass
        ``settings.gesture.pointer`` to enable the full filter stack.
        """
        if cfg is None:
            # No filters: raw sensitivity-only mode (default for tests and
            # headless / legacy callers that don't pass a config).
            return

        oe = cfg.one_euro
        if oe.enabled:
            self._oe_x = OneEuroFilter(
                min_cutoff=oe.min_cutoff,
                beta=oe.beta,
                d_cutoff=oe.d_cutoff,
            )
            self._oe_y = OneEuroFilter(
                min_cutoff=oe.min_cutoff,
                beta=oe.beta,
                d_cutoff=oe.d_cutoff,
            )

        k = cfg.kalman
        if k.enabled:
            self._kalman = KalmanFilter2D(
                process_noise=k.process_noise,
                measurement_noise_base=k.measurement_noise_base,
            )

        dz = cfg.dead_zone
        if dz.enabled:
            self._dead_zone = DeadZone(
                radius_px=dz.radius_px,
                hysteresis_px=dz.hysteresis_px,
            )

        ac = cfg.acceleration
        if ac.enabled:
            self._accel = AccelerationCurve(
                v_low=ac.v_low,
                v_high=ac.v_high,
                gain_low=ac.gain_low,
                gain_high=ac.gain_high,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def desktop(self) -> VirtualDesktop:
        """The virtual-desktop geometry used for mapping."""
        return self._desktop

    @property
    def sensitivity(self) -> float:
        """Current sensitivity multiplier."""
        return self._sensitivity

    # ------------------------------------------------------------------
    # Core mapping
    # ------------------------------------------------------------------

    def map(
        self,
        lm_x: float,
        lm_y: float,
        dt: float | None = None,
        confidence: float = 1.0,
    ) -> tuple[int, int]:
        """Convert normalized landmark coordinates to screen pixel coordinates.

        Filter pipeline applied in order:

        1. One-Euro filter (per-axis, adaptive cutoff based on velocity).
        2. Kalman 2-D (constant-velocity; updates measurement noise from *confidence*).
        3. Sensitivity scaling + clamp to virtual desktop.
        4. Dead-zone (operates in pixel space for accurate radius interpretation).
        5. Acceleration curve (also in pixel space).

        Args:
            lm_x:       Normalized x coordinate ∈ [0.0, 1.0] (from MediaPipe).
            lm_y:       Normalized y coordinate ∈ [0.0, 1.0].
            dt:         Time since last frame in seconds.  Auto-computed from
                        ``time.monotonic()`` when *None*.
            confidence: MediaPipe hand detection confidence ∈ [0, 1].  Used to
                        weight the Kalman measurement noise.

        Returns:
            ``(screen_x, screen_y)`` in virtual-desktop pixel coordinates.
        """
        now = time.monotonic()
        if dt is None:
            dt = max(now - self._prev_ts, 1e-6)
        self._prev_ts = now

        nx, ny = lm_x, lm_y

        # 1. One-Euro filter on normalised coords (removes sensor jitter).
        if self._oe_x is not None and self._oe_y is not None:
            nx = self._oe_x.filter(nx, timestamp=now)
            ny = self._oe_y.filter(ny, timestamp=now)

        # 2. Sensitivity scaling + clamp (before Kalman so we operate in
        #    pixel space where velocity is more intuitive to tune).
        s = self._sensitivity
        nx_s = max(0.0, min(1.0, (nx - 0.5) * s + 0.5))
        ny_s = max(0.0, min(1.0, (ny - 0.5) * s + 0.5))
        d = self._desktop
        px = d.left + nx_s * d.width
        py = d.top + ny_s * d.height

        # 3. Kalman filter in pixel space.
        if self._kalman is not None:
            px, py = self._kalman.update(px, py, confidence=confidence)

        # 4. Dead-zone (suppress micro-jitter near stationary position).
        if self._dead_zone is not None:
            px, py = self._dead_zone.apply(px, py)

        # 5. Acceleration curve — compute velocity from previous position.
        if self._accel is not None and self._prev_nx is not None:
            # Velocity in normalised units / second.
            vnx = (nx - self._prev_nx) / max(dt, 1e-6)
            vny = (ny - self._prev_ny) / max(dt, 1e-6)  # type: ignore[operator]
            import math

            v = math.sqrt(vnx * vnx + vny * vny)
            gain = self._accel.gain(v)
            # Apply gain as additive boost around the dead-zone-stabilised anchor.
            if self._dead_zone is not None and self._dead_zone._anchor_x is not None:
                ddx = px - self._dead_zone._anchor_x
                ddy = py - self._dead_zone._anchor_y  # type: ignore[operator]
                px = self._dead_zone._anchor_x + ddx * gain
                py = self._dead_zone._anchor_y + ddy * gain  # type: ignore[operator]

        self._prev_nx = nx
        self._prev_ny = ny

        sx = int(max(d.left, min(d.left + d.width, px)))
        sy = int(max(d.top, min(d.top + d.height, py)))
        return (sx, sy)

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all filter states (call when hand tracking is lost)."""
        if self._oe_x is not None:
            self._oe_x.reset()
        if self._oe_y is not None:
            self._oe_y.reset()
        if self._kalman is not None:
            self._kalman.reset()
        if self._dead_zone is not None:
            self._dead_zone.reset()
        self._prev_nx = None
        self._prev_ny = None
