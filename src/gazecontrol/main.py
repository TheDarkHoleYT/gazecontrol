"""GazeControl - Entry Point
Pipeline: EyeTrax (gaze) + MediaPipe (gesture) → Intent FSM → Window Manager.

Enterprise mode:
  - Gaze ensemble: TinyMLP 256→128→64 (30%) + L2CS-Net CNN (70%)
  - Smoothing: One Euro Filter (adattivo: smooth in fixation, reattivo in saccade)
  - Fixation Detection: I-VT per classificare fixation/saccade/pursuit
  - Drift Correction: edge snapping + implicit recalibration da azioni utente
  - Frame Preprocessing: CLAHE + quality score
  - Camera: 1280x720 (fallback 640x480)
  - Thread safety: lock sull'overlay data
  - Requisito: models/l2cs_net_gaze360.onnx (eseguire tools/download_l2cs.py)
"""
import os

os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import argparse
import logging
import sys
import threading
import time

import cv2
from eyetrax import GazeEstimator
from eyetrax.calibration import run_dense_grid_calibration
from eyetrax.calibration.adaptive import run_adaptive_calibration

import gazecontrol.config as config
from gazecontrol.capture.frame_grabber import FrameGrabber
from gazecontrol.capture.frame_preprocessor import FramePreprocessor
from gazecontrol.gaze.drift_corrector import DriftCorrector
from gazecontrol.gaze.eyetrax_patches import PatchError, apply_patches
from gazecontrol.gaze.face_crop import FaceCropper
from gazecontrol.gaze.fixation_detector import FixationDetector
from gazecontrol.gaze.gaze_mapper import GazeMapper
from gazecontrol.gaze.l2cs_model import L2CSModel
from gazecontrol.gaze.one_euro_filter import OneEuroFilter
from gazecontrol.gesture.feature_extractor import GestureFeatureExtractor
from gazecontrol.gesture.hand_detector import HandDetector
from gazecontrol.gesture.mlp_classifier import MLPClassifier
from gazecontrol.gesture.rule_classifier import RuleClassifier
from gazecontrol.intent.state_machine import IntentStateMachine
from gazecontrol.intent.window_selector import WindowSelector
from gazecontrol.overlay.overlay_window import OverlayWindow
from gazecontrol.utils.profiler import PipelineProfiler
from gazecontrol.window_manager.windows_mgr import WindowsManager

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def _open_camera_dshow(index: int = 0) -> cv2.VideoCapture:
    """Patch per Windows: apri camera con DirectShow (CAP_DSHOW) e configurazione completa.

    Replica la stessa configurazione di FrameGrabber (risoluzione, auto-exposure, warmup)
    per garantire frame stabili e ben esposti prima di passarli a MediaPipe.

    Prova tutti i candidati in ordine e restituisce il primo che si apre correttamente.
    Ogni tentativo fallito rilascia il VideoCapture prima di passare al successivo.
    """
    candidates = [
        (index, cv2.CAP_DSHOW),
        (index, cv2.CAP_ANY),
    ]
    if index != 0:
        candidates += [(0, cv2.CAP_DSHOW), (0, cv2.CAP_ANY)]
    else:
        candidates += [(1, cv2.CAP_DSHOW), (1, cv2.CAP_ANY)]

    for cam_index, backend in candidates:
        cap = cv2.VideoCapture(cam_index, backend)
        if cap.isOpened():
            break
        cap.release()
    else:
        raise RuntimeError(f"cannot open camera {index} (tried all backends)")

    # Configurazione identica a FrameGrabber per frame stabili
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # disabilita auto-exposure (riduce flicker)

    # Warmup: scarta i primi 5 frame (DirectShow restituisce spesso frame neri all'inizio)
    for _ in range(5):
        cap.read()

    return cap


def _wait_for_face_patched(cap, gaze_estimator, sw, sh, dur: int = 2) -> bool:
    """Versione migliorata di wait_for_face_and_countdown.

    Differenze rispetto all'originale eyetrax:
    - Sfondo grigio scuro (30,30,30) invece di nero puro: evita che la webcam
      aumenti l'exposure a dismisura vedendo uno schermo completamente nero
    - Preview camera nell'angolo in basso a destra: permette di verificare
      visivamente che la webcam stia riprendendo la faccia
    - Timeout di 60s con messaggio di errore più chiaro
    """
    import numpy as _np

    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fd_start = None
    countdown = False
    attempt_start = time.time()
    MAX_WAIT_S = 60

    _consecutive_read_failures = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            _consecutive_read_failures += 1
            if _consecutive_read_failures >= 30:
                logger.error("Camera stopped returning frames after %d consecutive failures", _consecutive_read_failures)
                return False
            if time.time() - attempt_start > MAX_WAIT_S:
                logger.error("Timeout waiting for valid frame from camera")
                return False
            continue
        _consecutive_read_failures = 0

        f, blink = gaze_estimator.extract_features(frame)
        face = f is not None and not blink

        # Sfondo grigio scuro invece di nero: la webcam non va in over-exposure
        canvas = _np.full((sh, sw, 3), 30, dtype=_np.uint8)

        # Preview camera (320x180) in basso a destra
        try:
            preview = cv2.resize(frame, (320, 180))
            canvas[sh - 190:sh - 10, sw - 330:sw - 10] = preview
            # Bordo attorno al preview
            cv2.rectangle(canvas, (sw - 331, sh - 191), (sw - 9, sh - 9),
                          (0, 200, 0) if face else (0, 0, 200), 2)
        except Exception:
            pass

        now = time.time()
        elapsed_total = now - attempt_start

        if face:
            if not countdown:
                fd_start = now
                countdown = True
            elapsed = now - fd_start
            if elapsed >= dur:
                return True
            t = elapsed / dur
            e = t * t * (3 - 2 * t)
            ang = 360 * (1 - e)
            cv2.ellipse(canvas, (sw // 2, sh // 2), (50, 50),
                        0, -90, -90 + ang, (0, 255, 0), -1)
            cv2.putText(canvas, "Faccia rilevata - resta fermo",
                        (sw // 2 - 280, sh // 2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            countdown = False
            fd_start = None

            if elapsed_total > MAX_WAIT_S:
                logger.error(
                    "Timeout face detection dopo %.0fs. "
                    "Controlla illuminazione e posizione davanti alla webcam.", elapsed_total
                )
                return False

            txt = "Faccia non rilevata"
            fs, thick = 2, 3
            size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            tx = (sw - size[0]) // 2
            ty = (sh + size[1]) // 2 - 60
            cv2.putText(canvas, txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), thick)

            hint = f"Assicurati di essere davanti alla webcam ({int(MAX_WAIT_S - elapsed_total)}s)"
            cv2.putText(canvas, hint, (sw // 2 - 350, ty + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1)

        cv2.imshow("Calibration", canvas)
        if cv2.waitKey(1) == 27:
            return False


def run_calibration(profile_name: str = 'default', adaptive: bool = False) -> None:
    """Avvia la calibrazione EyeTrax con TinyMLP e salva il profilo.

    adaptive=False  →  griglia densa 5×5 (25 punti, ~50 secondi)
    adaptive=True   →  9 punti base + 60 punti random blue-noise con
                       retraining incrementale ogni 10 (~2.5 minuti, massima precisione)
    """
    # Patch eyetrax internals: force CAP_DSHOW on Windows + improved face-wait UI.
    # Isolated in eyetrax_patches.py for version-safety and testability.
    try:
        apply_patches(_open_camera_dshow, _wait_for_face_patched)
    except PatchError as exc:
        logger.warning("eyetrax patch failed: %s — calibration may hang on Windows", exc)

    profile_path = os.path.join(config.PROFILES_DIR, f'{profile_name}.pkl')
    os.makedirs(config.PROFILES_DIR, exist_ok=True)

    # TinyMLP 256→128→64: non-lineare, ~10× più accurato di Ridge su questo task
    estimator = GazeEstimator(
        model_name='tiny_mlp',
        model_kwargs={
            'hidden_layer_sizes': (256, 128, 64),
            'max_iter': 1000,
            'alpha': 1e-4,
            'early_stopping': True,
        },
    )

    if adaptive:
        logger.info("Avvio calibrazione ADATTIVA (9 + 60 punti random)...")
        run_adaptive_calibration(estimator, num_random_points=60, retrain_every=10)
    else:
        logger.info("Avvio calibrazione griglia densa 5×5 (25 punti)...")
        run_dense_grid_calibration(estimator, rows=5, cols=5, order='serpentine')
    estimator.save_model(profile_path)
    logger.info("Profilo '%s' salvato in %s", profile_name, profile_path)


class GazeControlPipeline:
    """Orchestra tutti i moduli della pipeline in un loop a ~30fps."""

    def __init__(self, profile_name: str = 'default', show_overlay: bool = True) -> None:
        self.profile_name = profile_name
        self.show_overlay = show_overlay
        self._running = False

        # GazeEstimator DEVE essere creato nel thread che chiama extract_features.
        self.estimator = None
        self._is_calibrated = False

        self._init_screen()    # sets self._screen_w, self._screen_h
        self._init_capture()   # sets grabber, preprocessor
        self._init_gaze()      # sets gaze filters, drift, mapper, l2cs, face_cropper
        self._init_gesture()   # sets hand_detector, feature_extractor, classifiers
        self._init_intent()    # sets state_machine, window_selector, window_manager
        self._init_overlay()   # sets overlay

        # FPS e performance profiler
        self._fps_counter = 0
        self._fps_timer = time.time()
        self._actual_fps = 0.0
        self._profiler = PipelineProfiler(log_every_n=300)  # ogni ~10s

    # ------------------------------------------------------------------
    # Init helpers — each initialises a single subsystem
    # ------------------------------------------------------------------

    def _init_screen(self) -> None:
        """Detect screen dimensions with DPI awareness."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            self._screen_w = user32.GetSystemMetrics(0)
            self._screen_h = user32.GetSystemMetrics(1)
        except Exception:
            self._screen_w, self._screen_h = 1920, 1080
        logger.debug("Screen: %dx%d", self._screen_w, self._screen_h)

    def _init_capture(self) -> None:
        """Set up webcam grabber and frame preprocessor."""
        self.grabber = FrameGrabber()
        self._preprocessor = FramePreprocessor()

    def _init_gaze(self) -> None:
        """Set up gaze filters, drift corrector, mapper, CNN model."""
        self._filter_x = OneEuroFilter(
            freq=config.CAMERA_FPS,
            min_cutoff=config.GAZE_1EURO_MIN_CUTOFF,
            beta=config.GAZE_1EURO_BETA,
        )
        self._filter_y = OneEuroFilter(
            freq=config.CAMERA_FPS,
            min_cutoff=config.GAZE_1EURO_MIN_CUTOFF,
            beta=config.GAZE_1EURO_BETA,
        )
        self._fixation_detector = FixationDetector()
        self._drift_corrector = DriftCorrector(
            screen_w=self._screen_w,
            screen_h=self._screen_h,
        )
        self._gaze_mapper = GazeMapper(
            screen_w=self._screen_w,
            screen_h=self._screen_h,
        )
        self._l2cs = L2CSModel(config.L2CS_MODEL_PATH)
        self._face_cropper = FaceCropper()

        # Blink-hold state
        self._last_valid_gaze: tuple[int, int] | None = None
        self._blink_start: float | None = None

    def _init_gesture(self) -> None:
        """Set up hand detector, feature extractor and gesture classifiers."""
        self.hand_detector = HandDetector()
        self.feature_extractor = GestureFeatureExtractor()
        self.rule_classifier = RuleClassifier()
        self.mlp_classifier = MLPClassifier()

    def _init_intent(self) -> None:
        """Set up intent state-machine, window selector and window manager."""
        self.state_machine = IntentStateMachine()
        self.window_selector = WindowSelector()
        self.window_manager = WindowsManager()

    def _init_overlay(self) -> None:
        """Create the HUD overlay (skipped when show_overlay=False)."""
        self.overlay = OverlayWindow() if self.show_overlay else None

    # ------------------------------------------------------------------

    def load_profile(self) -> bool:
        """Chiamato DENTRO il pipeline thread dopo aver creato self.estimator."""
        profile_path = os.path.join(config.PROFILES_DIR, f'{self.profile_name}.pkl')
        if not os.path.exists(profile_path):
            logger.warning("Profilo '%s' non trovato. Eseguire --calibrate prima.", self.profile_name)
            return False
        try:
            self.estimator.load_model(profile_path)
            self._is_calibrated = True
            self._drift_corrector.reset()
            logger.info("Profilo '%s' caricato", self.profile_name)
            return True
        except Exception:
            logger.exception("Errore caricamento profilo '%s'", self.profile_name)
            return False

    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self.grabber.start():
            logger.error("Impossibile avviare la camera.")
            return

        self._running = True
        logger.info("Pipeline GazeControl Enterprise avviata")

        if self.overlay and self.show_overlay:
            self._start_with_qt()
        else:
            try:
                self._main_loop()
            except KeyboardInterrupt:
                logger.info("Interruzione utente")
            finally:
                self.stop()

    def _start_with_qt(self) -> None:
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)
        self.overlay.create_widget()

        pipeline_thread = threading.Thread(target=self._main_loop, daemon=True)
        pipeline_thread.start()

        interrupt_timer = QTimer()
        interrupt_timer.timeout.connect(self._check_interrupt)
        interrupt_timer.start(200)

        try:
            app.exec()
        finally:
            self.stop()

    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        """Loop principale ~30fps con enterprise gaze pipeline."""
        # Inizializza EyeTrax IN QUESTO THREAD (thread safety: MediaPipe VIDEO mode)
        try:
            self.estimator = GazeEstimator(
                model_name='tiny_mlp',
                model_kwargs={
                    'hidden_layer_sizes': (256, 128, 64),
                    'max_iter': 1000,
                    'alpha': 1e-4,
                    'early_stopping': True,
                },
            )
            logger.info("GazeEstimator (TinyMLP 256→128→64) inizializzato")
            self.load_profile()
        except Exception:
            logger.exception("Errore inizializzazione GazeEstimator")

        blink_hold_max_s = 0.4

        while self._running:
            t0 = time.monotonic()

            # 1. Cattura frame
            ok_bgr, frame_bgr = self.grabber.read_bgr()
            ok_rgb, frame_rgb = self.grabber.read()
            if not ok_bgr or not ok_rgb:
                continue

            # 2. Preprocessing del frame BGR (CLAHE + quality score)
            with self._profiler.stage('preprocess'):
                enhanced_bgr, quality = self._preprocessor.process(frame_bgr)

            # 3. Gaze — EyeTrax su frame enhanced
            gaze_point = None
            gaze_event = None

            if self._is_calibrated and self.estimator is not None:
                try:
                    with self._profiler.stage('landmarks'):
                        features, blink = self.estimator.extract_features(enhanced_bgr)

                    if blink:
                        # Blink hold: mantieni l'ultimo gaze valido per un breve periodo
                        if self._blink_start is None:
                            self._blink_start = t0
                        blink_duration = t0 - self._blink_start
                        if blink_duration < blink_hold_max_s and self._last_valid_gaze:
                            gaze_point = self._last_valid_gaze
                        # Reset filtri se blink lungo
                        if blink_duration > blink_hold_max_s:
                            self._filter_x.reset()
                            self._filter_y.reset()
                    elif features is not None and quality.is_usable:
                        self._blink_start = None
                        px, py = self.estimator.predict([features])[0]

                        # 3a. L2CS-Net ensemble (appearance-based CNN)
                        with self._profiler.stage('l2cs'):
                            face_crop = self._face_cropper.crop_from_frame(enhanced_bgr)
                            l2cs_angles = self._l2cs.predict(face_crop)
                        if l2cs_angles is not None:
                            yaw, pitch = l2cs_angles
                            l2cs_xy = self._gaze_mapper.predict(yaw, pitch)
                            if l2cs_xy is not None:
                                lw = config.GAZE_ENSEMBLE_LANDMARK_WEIGHT
                                aw = config.GAZE_ENSEMBLE_APPEARANCE_WEIGHT
                                px = lw * px + aw * l2cs_xy[0]
                                py = lw * py + aw * l2cs_xy[1]

                        # 3b. One Euro Filter (sostituisce EMA)
                        fx = self._filter_x.filter(float(px), timestamp=t0)
                        fy = self._filter_y.filter(float(py), timestamp=t0)

                        # 3c. Drift correction
                        cx, cy = self._drift_corrector.correct(fx, fy)

                        # 3d. Fixation detection
                        gaze_event = self._fixation_detector.update(cx, cy, t0)

                        # Durante saccade usa il punto raw corretto (più reattivo)
                        # Durante fixation usa centroide se disponibile (più accurato)
                        if gaze_event.type == 'saccade':
                            gaze_point = (int(cx), int(cy))
                        elif gaze_event.centroid:
                            gx, gy = gaze_event.centroid
                            gaze_point = (int(gx), int(gy))
                        else:
                            gaze_point = (int(cx), int(cy))

                        self._last_valid_gaze = gaze_point
                    elif features is not None and not quality.is_usable:
                        # Frame di bassa qualità (sfocato/overexposed): hold
                        gaze_point = self._last_valid_gaze

                except Exception:
                    logger.warning("EyeTrax predict fallito", exc_info=True)

            # 4. Hand Gesture
            with self._profiler.stage('gesture'):
                hand_result = self.hand_detector.process(frame_rgb)
            gesture_id = None
            gesture_confidence = 0.0
            hand_position = None

            if hand_result:
                feat = self.feature_extractor.extract(hand_result)
                if feat:
                    hand_position = (
                        feat['wrist_x'] * self._screen_w * config.DRAG_HAND_SENSITIVITY,
                        feat['wrist_y'] * self._screen_h * config.DRAG_HAND_SENSITIVITY,
                    )
                gesture_id, gesture_confidence = self.rule_classifier.classify(feat)
                if gesture_id is None and self.mlp_classifier.is_loaded():
                    gesture_id, gesture_confidence = self.mlp_classifier.classify(feat)

            # 5. Intent Engine
            target_window = None
            if gaze_point:
                target_window = self.window_selector.find_window(gaze_point)
            action = self.state_machine.update(
                gaze_point=gaze_point,
                target_window=target_window,
                gesture_id=gesture_id,
                gesture_confidence=gesture_confidence,
                hand_position=hand_position,
            )

            # 6. Window Manager + Drift feedback
            if action:
                if action.get('type') == 'CLOSE_APP':
                    logger.info("Doppio pinch: chiusura GazeControl")
                    self._running = False
                else:
                    self.window_manager.execute(action)
                    # Implicit recalibration drift: usa azione utente come ground truth
                    if target_window and gaze_point and action.get('type') in (
                        'DRAG', 'CLOSE', 'MINIMIZE', 'MAXIMIZE', 'BRING_FRONT'
                    ):
                        self._drift_corrector.on_action(gaze_point, target_window)

            # 7. Overlay HUD
            if self.overlay:
                self.overlay.update(
                    gaze_point=gaze_point,
                    state=self.state_machine.state,
                    target_window=target_window,
                    gesture_id=gesture_id,
                    gesture_confidence=gesture_confidence,
                    is_calibrated=self._is_calibrated,
                    gaze_event_type=gaze_event.type if gaze_event else None,
                )

            # Profiler tick
            self._profiler.tick()

            # FPS
            self._fps_counter += 1
            elapsed = time.time() - self._fps_timer
            if elapsed >= 1.0:
                self._actual_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_timer = time.time()
                logger.debug(
                    "FPS: %.1f | State: %s | Gaze: %s | Quality: %.0f | "
                    "Drift: (%.0f, %.0f)",
                    self._actual_fps,
                    self.state_machine.state,
                    gaze_point,
                    quality.laplacian_var,
                    self._drift_corrector.offset[0],
                    self._drift_corrector.offset[1],
                )

            loop_time = time.monotonic() - t0
            sleep_time = (1.0 / config.CAMERA_FPS) - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------

    def _check_interrupt(self) -> None:
        if not self._running:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                app.quit()

    def stop(self) -> None:
        self._running = False
        self.grabber.stop()
        self.hand_detector.close()
        if self.overlay:
            self.overlay.stop()
        logger.info("Pipeline GazeControl fermata")


# ------------------------------------------------------------------

def main() -> None:
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser(
        description='GazeControl Enterprise - Desktop control via Eye Tracking & Gestures'
    )
    parser.add_argument('--calibrate', action='store_true',
                        help='Calibrazione griglia 5×5 (25 punti, ~50s) con TinyMLP')
    parser.add_argument('--adaptive', action='store_true',
                        help='Calibrazione adattiva 9+60 punti (massima precisione, ~2.5min)')
    parser.add_argument('--profile', default='default', help='Nome profilo calibrazione')
    parser.add_argument('--no-overlay', action='store_true', help='Disabilita overlay HUD')
    args = parser.parse_args()

    os.makedirs(config.PROFILES_DIR, exist_ok=True)

    if args.calibrate or args.adaptive:
        run_calibration(profile_name=args.profile, adaptive=args.adaptive)
    else:
        pipeline = GazeControlPipeline(
            profile_name=args.profile,
            show_overlay=not args.no_overlay,
        )
        pipeline.start()


if __name__ == '__main__':
    main()
