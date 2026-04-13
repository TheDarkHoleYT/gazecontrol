"""
GazeControl - Parametri Globali di Configurazione
"""
import os

_HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(_HERE, '..', 'models')

# =============================================================================
# CAMERA
# =============================================================================
CAMERA_INDEX = 0
# 1280x720 preferito; frame_grabber fa fallback a 640x480 se non supportato.
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
CAMERA_FPS   = 30

# =============================================================================
# GAZE (Enterprise)
# =============================================================================
# Modalità modello gaze: 'landmark' | 'appearance' | 'hybrid'
GAZE_MODEL_MODE = 'hybrid'

# One Euro Filter (sostituisce EMA).
# min_cutoff: frequenza di taglio minima durante le fixation (Hz) — valori più
#   alti = meno smooth, più reattivo anche a riposo.
# beta: quanto aumenta la cutoff con la velocità — valori più alti = più
#   reattivo durante le saccadi.
GAZE_1EURO_MIN_CUTOFF = 1.5
GAZE_1EURO_BETA       = 0.007

# Soglia confidence: sotto questa soglia il frame gaze viene scartato.
GAZE_CONFIDENCE_THRESHOLD = 0.6

# Cartella profili calibrazione
PROFILES_DIR = os.path.join(_HERE, '..', 'profiles')

# Modello L2CS-Net (appearance-based CNN, ONNX)
L2CS_MODEL_PATH = os.path.join(MODELS_DIR, 'l2cs_net_gaze360.onnx')

# Pesi ensemble landmark vs. appearance
GAZE_ENSEMBLE_LANDMARK_WEIGHT   = 0.3
GAZE_ENSEMBLE_APPEARANCE_WEIGHT = 0.7

# =============================================================================
# HAND TRACKING
# =============================================================================
HAND_MAX_HANDS = 1
HAND_MIN_DETECTION_CONFIDENCE = 0.7
HAND_MIN_TRACKING_CONFIDENCE  = 0.5

# Risoluzione di riferimento usata per scalare le velocità gesture nel feature extractor.
# Le soglie come SWIPE_VELOCITY_THRESHOLD sono calibrate su questa risoluzione.
# Aggiornare entrambi se si cambia la risoluzione della camera.
FEATURE_REF_WIDTH  = FRAME_WIDTH   # 1280
FEATURE_REF_HEIGHT = FRAME_HEIGHT  # 720

# Soglie gesture
SWIPE_VELOCITY_THRESHOLD = 200   # px/s (alla risoluzione FEATURE_REF_WIDTH x FEATURE_REF_HEIGHT)

# Sensibilità drag mano → finestra.
# 1.0 = 1:1 (tutta larghezza camera = tutta larghezza schermo)
# 1.5 = default (più reattivo)
DRAG_HAND_SENSITIVITY  = 1.5
# Sensibilità resize mano → finestra (px schermo per px mano)
RESIZE_HAND_SENSITIVITY = 2.0

# =============================================================================
# CLASSIFICATORE GESTURE
# =============================================================================
GESTURE_CONFIDENCE_THRESHOLD = 0.85
GESTURE_LABELS = [
    'GRAB', 'RELEASE', 'PINCH',
    'SWIPE_LEFT', 'SWIPE_RIGHT', 'CLOSE_SIGN',
    'SCROLL_UP', 'SCROLL_DOWN', 'MAXIMIZE',
]
MLP_MODEL_PATH = os.path.join(MODELS_DIR, 'gesture_mlp.onnx')

# =============================================================================
# INTENT ENGINE / STATE MACHINE
# =============================================================================
DWELL_TIME_MS   = 400    # ms sguardo fisso per selezionare finestra
READY_TIMEOUT_S = 3.0    # s max in stato READY senza gesture
COOLDOWN_MS     = 300    # ms post-azione

# =============================================================================
# OVERLAY HUD
# =============================================================================
OVERLAY_GAZE_DOT_RADIUS  = 8
OVERLAY_GAZE_DOT_COLOR   = (0, 220, 0)
OVERLAY_TARGETING_COLOR  = (0, 100, 255)
OVERLAY_READY_COLOR      = (255, 140, 0)

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = 'INFO'
LOG_FILE  = os.path.join(_HERE, '..', 'gazecontrol.log')
