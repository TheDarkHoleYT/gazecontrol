"""
GazeControl - Frame Grabber
Gestione webcam, buffering e pre-processing dei frame.
"""
import cv2
import numpy as np
import threading
import logging
from typing import Optional, Tuple
import gazecontrol.config as config

logger = logging.getLogger(__name__)


class FrameGrabber:
    """
    Cattura frame dalla webcam in un thread separato per evitare blocchi
    sulla pipeline principale. Espone sempre l'ultimo frame disponibile.
    """

    def __init__(self, camera_index: int = config.CAMERA_INDEX,
                 width: int = config.FRAME_WIDTH,
                 height: int = config.FRAME_HEIGHT,
                 fps: int = config.CAMERA_FPS):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0

    def start(self) -> bool:
        """Apre la camera e avvia il thread di cattura. Ritorna True se OK."""
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback senza CAP_DSHOW
            self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.error(f"Impossibile aprire camera index={self.camera_index}")
            return False

        # Tenta la risoluzione preferita; se la camera non la supporta, usa quella reale.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Disabilita auto-exposure se possibile (riduce varianza brightness)
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != self.width or actual_h != self.height:
            logger.warning(
                "Risoluzione richiesta %dx%d non supportata dalla camera. "
                "Uso %dx%d.", self.width, self.height, actual_w, actual_h
            )
            self.width = actual_w
            self.height = actual_h

        # Leggi un frame di test
        ret, frame = self._cap.read()
        if not ret:
            logger.error("Camera aperta ma impossibile leggere frame")
            return False

        with self._lock:
            self._frame = frame

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("FrameGrabber avviato: %dx%d@%dfps", self.width, self.height, self.fps)
        return True

    def _capture_loop(self):
        """Loop di cattura eseguito nel thread separato."""
        consecutive_drops = 0
        max_drops = 30  # ~1s a 30fps prima di tentare il restart

        while self._running:
            ret, frame = self._cap.read()
            if ret:
                consecutive_drops = 0
                with self._lock:
                    self._frame = frame
                    self._frame_count += 1
            else:
                consecutive_drops += 1
                if consecutive_drops == 1:
                    logger.warning("Frame drop rilevato")
                if consecutive_drops >= max_drops:
                    logger.error("Camera persa (%d drop consecutivi). Tentativo restart...",
                                 consecutive_drops)
                    self._restart_camera()
                    consecutive_drops = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Ritorna (success, frame_preprocessato).
        Il frame è già in RGB, flippato orizzontalmente e ridimensionato.
        """
        with self._lock:
            if self._frame is None:
                return False, None
            frame = self._frame.copy()

        frame = self._preprocess(frame)
        return True, frame

    def read_bgr(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Ritorna il frame grezzo in BGR (per visualizzazione OpenCV)."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Pre-processing standard:
        1. Flip orizzontale (mirror naturale)
        2. Resize se necessario
        3. Conversione BGR -> RGB (richiesto da MediaPipe)
        """
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _restart_camera(self):
        """Tenta di riaprire la camera dopo disconnessione (BUG-5: usa self._lock)."""
        import time
        with self._lock:
            try:
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
            except Exception:
                pass
        time.sleep(1.0)
        for _ in range(3):
            try:
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(self.camera_index)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.fps)
                    with self._lock:
                        self._cap = cap
                    logger.info("Camera riaperta con successo")
                    return
                cap.release()
            except Exception:
                pass
            time.sleep(2.0)
        logger.error("Impossibile riaprire la camera dopo 3 tentativi")

    def stop(self):
        """Ferma il thread e rilascia la camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        logger.info("FrameGrabber fermato")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def actual_resolution(self) -> Tuple[int, int]:
        with self._lock:
            if self._cap is None:
                return (0, 0)
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
