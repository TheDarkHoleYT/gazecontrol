"""L2CS-Net ONNX Wrapper — modello appearance-based per gaze estimation.

L2CS-Net (Abdelrahman & Hossny, 2022) predice yaw e pitch come combinazione di:
  - Classificazione su 90 bin angolari (softmax)
  - Regressione fine dell'offset residuo

L'output è (yaw, pitch) in gradi, dove:
  - yaw   : rotazione orizzontale (negativo = sinistra, positivo = destra)
  - pitch : rotazione verticale   (negativo = basso, positivo = alto)

Il modello ONNX si trova in models/l2cs_net_gaze360.onnx (~100 MB).
Se il file non esiste, questo modulo funziona in modalità "disabled" e
restituisce sempre None (il sistema usa solo il path eyetrax landmark-based).

Download e conversione: vedere tools/download_l2cs.py
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Range angolare del modello (training su Gaze360: ±180° yaw, ±90° pitch)
_NUM_BINS = 90
_ANGLE_RANGE_DEG = 180.0
_BIN_STEP = _ANGLE_RANGE_DEG / _NUM_BINS
_BIN_CENTERS = (np.arange(_NUM_BINS, dtype=np.float32) - _NUM_BINS / 2) * _BIN_STEP


class L2CSModel:
    """Wrapper ONNX per L2CS-Net.

    Args:
        model_path : percorso al file .onnx.
        providers  : provider ONNX Runtime (default: DirectML se disponibile, poi CPU).
    """

    def __init__(
        self,
        model_path: str,
        providers: list[str] | None = None,
    ) -> None:
        self._session = None
        self._input_name = None

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"L2CS-Net model non trovato: {model_path}\n"
                "Eseguire prima: python tools/download_l2cs.py"
            )

        try:
            import onnxruntime as ort

            if providers is None:
                available = ort.get_available_providers()
                # Preferisci GPU DirectML su Windows, poi CUDA, poi CPU
                for pref in ('DmlExecutionProvider', 'CUDAExecutionProvider',
                             'CPUExecutionProvider'):
                    if pref in available:
                        providers = [pref, 'CPUExecutionProvider']
                        break
                else:
                    providers = ['CPUExecutionProvider']

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                model_path, sess_options=so, providers=providers
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.info(
                "L2CS-Net caricato: %s | provider: %s",
                os.path.basename(model_path),
                self._session.get_providers()[0],
            )
        except ImportError as exc:
            raise ImportError(
                "onnxruntime non installato. "
                "Installare con: pip install onnxruntime-directml"
            ) from exc
        except Exception:
            logger.exception("Errore caricamento L2CS-Net da %s", model_path)

    @property
    def is_loaded(self) -> bool:
        """Return True when the ONNX inference session is ready."""
        return self._session is not None

    def predict(self, face_crop: np.ndarray) -> tuple[float, float] | None:
        """Predice yaw e pitch in GRADI dal crop del volto.

        Args:
            face_crop : numpy array (1, 3, 224, 224) float32.

        Returns:
            (yaw_deg, pitch_deg) oppure None se il modello non è caricato.
        """
        if self._session is None or face_crop is None:
            return None

        try:
            outputs = self._session.run(
                None, {self._input_name: face_crop}
            )
            # L2CS-Net output: [yaw_logits (1,90), pitch_logits (1,90)]
            # oppure [yaw_deg (1,1), pitch_deg (1,1)] a seconda della versione ONNX
            if len(outputs) == 2:
                out0, out1 = outputs[0], outputs[1]
                if out0.shape[-1] == _NUM_BINS or out0.shape[-1] > 2:
                    # Output logits su 90 bin → weighted average via softmax
                    yaw_deg   = self._bins_to_angle(out0[0])
                    pitch_deg = self._bins_to_angle(out1[0])
                else:
                    # Output scalare diretto
                    yaw_deg   = float(out0.flat[0])
                    pitch_deg = float(out1.flat[0])
            elif len(outputs) == 1:
                # Formato alternativo: singolo output (1, 2)
                yaw_deg   = float(outputs[0][0, 0])
                pitch_deg = float(outputs[0][0, 1])
            else:
                return None

            return yaw_deg, pitch_deg

        except Exception:
            logger.debug("L2CS-Net predict fallito", exc_info=True)
            return None

    @staticmethod
    def _bins_to_angle(logits: np.ndarray) -> float:
        """Converte logits di 90 bin in angolo in gradi via softmax + weighted mean."""
        exp = np.exp(logits - logits.max())
        softmax = exp / (exp.sum() + 1e-9)
        return float(np.dot(softmax, _BIN_CENTERS))
