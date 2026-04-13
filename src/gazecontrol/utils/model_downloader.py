"""
Scarica i modelli MediaPipe Tasks se non già presenti.
"""
import os
import urllib.request
import logging

logger = logging.getLogger(__name__)

MODELS = {
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    ),
    'hand_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    ),
}


def ensure_model(model_name: str, models_dir: str) -> str:
    """Ritorna il percorso del modello, scaricandolo se assente."""
    os.makedirs(models_dir, exist_ok=True)
    dest = os.path.join(models_dir, model_name)
    if os.path.exists(dest):
        return dest

    url = MODELS.get(model_name)
    if url is None:
        raise ValueError(f"Modello sconosciuto: {model_name}")

    logger.info(f"Download modello {model_name} da MediaPipe...")
    try:
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Modello salvato in {dest}")
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Impossibile scaricare {model_name}: {e}") from e

    return dest
