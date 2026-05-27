"""GazeStage — pipeline stage that runs a :class:`GazeBackend`.

The stage owns the backend (built via the runtime factory), the per-axis
:class:`OneEuroFilter` smoothing, the :class:`DriftCorrector`, and the
I-VT :class:`FixationDetector`. Its output is written to the shared
:class:`FrameContext` for downstream stages (PointerFusion, Interaction).

Heavy resources (ONNX, MediaPipe Face Mesh) are allocated in :meth:`start`
to honour the legacy single-thread MediaPipe contract.

Failure policy (G17, ADR-0008)
------------------------------
After ``FusionSettings.gaze_failure_threshold_frames`` consecutive
prediction failures (exception or ``None`` return) the stage takes one
of three actions per ``FusionSettings.gaze_failure_policy``:

* ``"continue"``  — legacy behaviour (no state change beyond logging).
* ``"hand_only"`` — flip :attr:`FrameContext.gaze_backend_down` so HUD /
  telemetry surface a degraded badge; PointerFusionStage already
  routes to the hand when ``gaze_screen is None``.
* ``"stop"``      — raise :class:`GazeBackendError` so the engine
  shuts down cleanly via its crash-handler hook.

The counter resets on the first successful prediction, emitting a
``gaze.recovered`` INFO log line on the transition.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from gazecontrol.errors import GazeBackendError
from gazecontrol.filters.one_euro import OneEuroFilter
from gazecontrol.gaze.backend import GazeBackend, GazePrediction
from gazecontrol.gaze.drift_corrector import DriftCorrector
from gazecontrol.gaze.fixation_detector import FixationDetector
from gazecontrol.pipeline.context import FrameContext
from gazecontrol.settings import AppSettings, get_settings

logger = logging.getLogger(__name__)

#: Optional callback the runtime can register on the stage to receive
#: "policy says stop" signals. Returns nothing — the engine is expected
#: to drive its own crash-handler shutdown sequence from inside.
StopCallback = Callable[[], None]


class GazeStage:
    """Pipeline stage producing per-frame gaze estimates."""

    name = "gaze"

    def __init__(
        self,
        backend: GazeBackend,
        screen_w: int,
        screen_h: int,
        settings: AppSettings | None = None,
        *,
        on_stop_requested: StopCallback | None = None,
    ) -> None:
        self._backend = backend
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._settings = settings
        self._on_stop_requested = on_stop_requested

        self._filter_x: OneEuroFilter | None = None
        self._filter_y: OneEuroFilter | None = None
        self._drift: DriftCorrector | None = None
        self._fixation: FixationDetector | None = None

        # Blink hold state.
        self._last_valid_xy: tuple[int, int] | None = None
        self._blink_started_at: float | None = None
        self._blink_hold_max_s: float = 0.4

        # G17 — failure policy state.
        self._consecutive_failures: int = 0
        self._failure_threshold: int = 10
        self._failure_policy: str = "hand_only"
        self._backend_down: bool = False
        self._stop_requested: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Build filters/drift/fixation and start the backend."""
        s = self._settings or get_settings()
        gcfg = s.gaze
        try:
            ok = self._backend.start()
        except Exception:
            logger.exception("GazeStage: backend start raised.")
            return False
        if not ok:
            logger.warning("GazeStage: backend %s declined to start.", self._backend.name)
            return False

        self._filter_x = OneEuroFilter(
            min_cutoff=gcfg.one_euro_min_cutoff,
            beta=gcfg.one_euro_beta,
        )
        self._filter_y = OneEuroFilter(
            min_cutoff=gcfg.one_euro_min_cutoff,
            beta=gcfg.one_euro_beta,
        )
        if gcfg.drift.enabled:
            self._drift = DriftCorrector(
                screen_w=self._screen_w,
                screen_h=self._screen_h,
                edge_margin_px=gcfg.drift.edge_margin_px,
                edge_correction_rate=gcfg.drift.edge_correction_rate,
                implicit_alpha=gcfg.drift.implicit_alpha,
                max_correction_px=float(gcfg.drift.max_correction_px),
                mode=gcfg.drift.mode,
                recenter_sample_count=gcfg.drift.recenter_sample_count,
                convergence_window=gcfg.drift.convergence_window,
                convergence_threshold_px=gcfg.drift.convergence_threshold_px,
                kalman_process_noise=gcfg.drift.kalman_process_noise,
                kalman_measurement_noise=gcfg.drift.kalman_measurement_noise,
            )
        self._fixation = FixationDetector(
            screen_px_per_degree=gcfg.px_per_deg,
            fixation_vel_thr=gcfg.fixation_velocity_deg_s,
            saccade_vel_thr=gcfg.saccade_velocity_deg_s,
        )
        self._blink_hold_max_s = gcfg.blink_hold_max_s
        # G17 — pick up fusion-side failure policy at start. Reading from
        # s.fusion keeps the gaze settings free of pipeline-flow concerns.
        self._failure_threshold = int(s.fusion.gaze_failure_threshold_frames)
        self._failure_policy = s.fusion.gaze_failure_policy
        self._consecutive_failures = 0
        self._backend_down = False
        self._stop_requested = False
        return True

    def stop(self) -> None:
        """Release the backend and reset filter/fixation state."""
        try:
            self._backend.stop()
        except Exception:
            logger.exception("GazeStage: backend stop raised.")
        self._filter_x = None
        self._filter_y = None
        self._drift = None
        self._fixation = None
        self._last_valid_xy = None
        self._blink_started_at = None

    def is_calibrated(self) -> bool:
        """True when the wrapped backend reports a usable calibration."""
        return self._backend.is_calibrated()

    @property
    def backend_name(self) -> str:
        """Name of the wrapped backend (for HUD diagnostics)."""
        return self._backend.name

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process(self, ctx: FrameContext) -> FrameContext:
        """Predict, filter, drift-correct, and classify one gaze sample."""
        if not ctx.capture_ok or ctx.frame_bgr is None or ctx.frame_rgb is None:
            return ctx

        try:
            prediction: GazePrediction | None = self._backend.predict(
                ctx.frame_bgr,
                ctx.frame_rgb,
                ctx.t0,
            )
        except Exception:
            logger.debug("GazeStage: backend predict raised.", exc_info=True)
            prediction = None

        if prediction is None:
            self._on_failure()
            ctx.face_present = False
            ctx.gaze_screen = self._apply_blink_hold(ctx.t0, hold_active=False)
            ctx.gaze_confidence = 0.0
            ctx.gaze_blink = False
            ctx.gaze_backend_down = self._backend_down
            return ctx

        # Successful sample — reset the failure counter + degrade state.
        self._on_recovery()
        ctx.gaze_backend_down = self._backend_down

        ctx.face_present = True
        ctx.gaze_yaw_pitch_deg = prediction.yaw_pitch_deg

        if prediction.blink:
            ctx.gaze_blink = True
            ctx.gaze_screen = self._apply_blink_hold(ctx.t0, hold_active=True)
            ctx.gaze_confidence = 0.0
            return ctx

        # Reset blink timer on a valid sample.
        self._blink_started_at = None
        ctx.gaze_blink = False

        raw_x, raw_y = prediction.screen_xy
        # Filter both axes (1€ adaptive low-pass).
        if self._filter_x is not None and self._filter_y is not None:
            fx = self._filter_x.filter(float(raw_x), timestamp=ctx.t0)
            fy = self._filter_y.filter(float(raw_y), timestamp=ctx.t0)
        else:
            fx, fy = float(raw_x), float(raw_y)

        if self._drift is not None:
            cx, cy = self._drift.correct(fx, fy)
        else:
            cx, cy = fx, fy

        if self._fixation is not None:
            event = self._fixation.update(cx, cy, ctx.t0)
            ctx.gaze_event = event
            if event.type == "fixation" and event.centroid is not None:
                gx, gy = event.centroid
            else:
                gx, gy = cx, cy
        else:
            gx, gy = cx, cy

        x = max(0, min(self._screen_w - 1, int(gx)))
        y = max(0, min(self._screen_h - 1, int(gy)))
        ctx.gaze_screen = (x, y)
        ctx.gaze_confidence = prediction.confidence
        self._last_valid_xy = (x, y)
        return ctx

    # ------------------------------------------------------------------
    # Failure policy (G17, ADR-0008)
    # ------------------------------------------------------------------

    def _on_failure(self) -> None:
        """Record a missing prediction and trigger the configured policy.

        ``self._consecutive_failures`` grows monotonically until a
        successful predict resets it. The configured policy fires
        exactly once per degradation event (the ``backend_down`` flag
        guards re-entry) so the HUD does not flicker between states.
        """
        self._consecutive_failures += 1
        if self._consecutive_failures < self._failure_threshold:
            return
        if self._backend_down:
            # Already degraded — keep state, no extra log spam.
            return
        self._backend_down = True
        if self._failure_policy == "continue":
            logger.info(
                "GazeStage: backend silent for %d frames — policy='continue', "
                "keeping legacy behaviour.",
                self._consecutive_failures,
            )
            return
        if self._failure_policy == "hand_only":
            logger.warning(
                "GazeStage: backend silent for %d frames — policy='hand_only', "
                "routing pointer to hand exclusively until recovery.",
                self._consecutive_failures,
            )
            return
        if self._failure_policy == "stop":
            logger.error(
                "GazeStage: backend silent for %d frames — policy='stop', "
                "requesting engine shutdown.",
                self._consecutive_failures,
            )
            self._stop_requested = True
            if self._on_stop_requested is not None:
                try:
                    self._on_stop_requested()
                except Exception:
                    logger.exception(
                        "GazeStage: on_stop_requested callback raised."
                    )
            raise GazeBackendError(
                f"Gaze backend silent for {self._consecutive_failures} frames "
                f"under policy='stop'."
            )

    def _on_recovery(self) -> None:
        """Reset failure state on a successful prediction.

        Emits a single ``gaze.recovered`` INFO log line on the
        backend_down → up transition so operators can find the recovery
        point in structured logs.
        """
        if self._consecutive_failures == 0 and not self._backend_down:
            return
        if self._backend_down:
            logger.info(
                "GazeStage: backend recovered after %d missed frames.",
                self._consecutive_failures,
            )
        self._consecutive_failures = 0
        self._backend_down = False

    @property
    def backend_down(self) -> bool:
        """True while the failure policy reports the backend as degraded."""
        return self._backend_down

    # ------------------------------------------------------------------
    # Drift feedback (called by ActionStage on user-initiated actions)
    # ------------------------------------------------------------------

    def on_user_action(
        self,
        gaze_point: tuple[int, int],
        target_rect: tuple[int, int, int, int],
    ) -> None:
        """Update drift estimate from a user action (implicit recalibration)."""
        if self._drift is None:
            return
        self._drift.on_action(
            (float(gaze_point[0]), float(gaze_point[1])),
            {"rect": target_rect},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_blink_hold(
        self,
        t: float,
        *,
        hold_active: bool,
    ) -> tuple[int, int] | None:
        """Return last valid gaze for up to ``blink_hold_max_s`` during blinks."""
        if not hold_active or self._last_valid_xy is None:
            return None
        if self._blink_started_at is None:
            self._blink_started_at = t
        if (t - self._blink_started_at) > self._blink_hold_max_s:
            return None
        return self._last_valid_xy
