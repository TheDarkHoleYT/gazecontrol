"""
Download e conversione di L2CS-Net in formato ONNX.

L2CS-Net (Abdelrahman & Hossny, 2022) -- licenza Apache 2.0.
Paper: https://arxiv.org/abs/2203.03339
Repo:  https://github.com/Ahmednull/L2CS-Net

I pesi sono su Google Drive. Lo script prova piu' metodi in ordine:
  1. gdown con fuzzy mode (gestisce quota e link condivisi)
  2. Istruzioni manuali chiare se tutto fallisce

Requisiti (solo per conversione, non per inference):
    pip install gdown torch torchvision onnx

Uso:
    python tools/download_l2cs.py
    python tools/download_l2cs.py --no-fp16        # mantieni FP32
    python tools/download_l2cs.py --skip-download  # se hai gia' il .pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Fix encoding su Windows (cp1252 non supporta caratteri box-drawing)
if sys.stdout.encoding and sys.stdout.encoding.lower() in ("cp1252", "cp850"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUT_PATH = MODELS_DIR / "l2cs_net_gaze360.onnx"
_WEIGHTS_FILE = MODELS_DIR / "L2CSNet_gaze360.pkl"

# Google Drive folder ufficiale L2CS-Net (dal README del repo)
# https://github.com/Ahmednull/L2CS-Net
_GDRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing"
)
_GDRIVE_FOLDER_ID = "17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd"

_MANUAL_MSG = """
=================================================================
  DOWNLOAD MANUALE necessario per i pesi L2CS-Net (Gaze360)
=================================================================

  1. Apri questa cartella Google Drive nel browser:
     https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd

  2. Scarica il file:  L2CSNet_gaze360.pkl  (~100 MB)

  3. Copialo in:
     {weights_path}

  4. Poi esegui di nuovo:
     python tools/download_l2cs.py --skip-download

=================================================================
"""


def _ensure_gdown() -> bool:
    try:
        import gdown  # noqa: F401

        return True
    except ImportError:
        print("[L2CS] Installo gdown...")
        import subprocess

        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "gdown", "-q"],
            capture_output=True,
        )
        return r.returncode == 0


def download_weights() -> bool:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if _WEIGHTS_FILE.exists() and _WEIGHTS_FILE.stat().st_size > 1_000_000:
        print(f"[L2CS] Pesi gia' presenti: {_WEIGHTS_FILE}")
        return True

    print("[L2CS] Download pesi L2CS-Net Gaze360 (~100 MB)...")
    print(f"       Destinazione: {_WEIGHTS_FILE}\n")

    if not _ensure_gdown():
        print("[ERRORE] Impossibile installare gdown.")
        print(_MANUAL_MSG.format(weights_path=_WEIGHTS_FILE))
        return False

    import gdown

    # Prova 1: scarica la cartella Drive e cerca L2CSNet_gaze360.pkl
    try:
        print("[L2CS] Tentativo 1: download cartella Google Drive...")
        tmp_dir = MODELS_DIR / "_l2cs_tmp"
        tmp_dir.mkdir(exist_ok=True)
        gdown.download_folder(
            _GDRIVE_FOLDER_URL,
            output=str(tmp_dir),
            quiet=False,
            use_cookies=False,
        )
        # Cerca il file pkl nella cartella scaricata
        for candidate in tmp_dir.rglob("*.pkl"):
            if "gaze360" in candidate.name.lower() or "l2cs" in candidate.name.lower():
                candidate.rename(_WEIGHTS_FILE)
                break
        # Pulizia tmp
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)

        if _WEIGHTS_FILE.exists() and _WEIGHTS_FILE.stat().st_size > 1_000_000:
            print(f"\n[L2CS] Download OK: {_WEIGHTS_FILE.stat().st_size / 1e6:.1f} MB")
            return True
    except Exception as e:
        print(f"\n[WARN] Tentativo 1 fallito: {e}")
        import shutil

        shutil.rmtree(MODELS_DIR / "_l2cs_tmp", ignore_errors=True)

    # Prova 2: fuzzy=True su URL cartella
    try:
        print("[L2CS] Tentativo 2: gdown fuzzy folder...")
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{_GDRIVE_FOLDER_ID}",
            output=str(MODELS_DIR),
            quiet=False,
            fuzzy=True,
        )
        if _WEIGHTS_FILE.exists() and _WEIGHTS_FILE.stat().st_size > 1_000_000:
            print(f"\n[L2CS] Download OK: {_WEIGHTS_FILE.stat().st_size / 1e6:.1f} MB")
            return True
    except Exception as e:
        print(f"\n[WARN] Tentativo 2 fallito: {e}")

    # Pulizia file parziali
    if _WEIGHTS_FILE.exists() and _WEIGHTS_FILE.stat().st_size < 1_000_000:
        _WEIGHTS_FILE.unlink()

    print("\n[ERRORE] Download automatico fallito (Google Drive limita l'accesso).")
    print(_MANUAL_MSG.format(weights_path=_WEIGHTS_FILE))
    return False


def build_l2cs_model():
    """
    Costruisce l'architettura L2CS-Net in PyTorch.

    Le chiavi del checkpoint sono flat (conv1, bn1, layer1..4, fc_yaw, fc_pitch),
    quindi i layer ResNet devono essere attributi diretti del modulo — NON wrapped
    in nn.Sequential (che produrrebbe chiavi 'backbone.0.weight' ecc.).
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError:
        print("[ERRORE] PyTorch non installato.")
        print("         Eseguire: pip install torch torchvision")
        sys.exit(1)

    class L2CS(nn.Module):
        """
        Architettura L2CS-Net Gaze360.

        Chiavi checkpoint (verificate sul .pkl ufficiale):
          conv1, bn1, layer1-4, avgpool  → standard ResNet50
          fc_yaw_gaze   : Linear(2048, 90) — 90 bin yaw
          fc_pitch_gaze : Linear(2048, 90) — 90 bin pitch
          fc_finetune   : Linear(2051, 3)  — 3D gaze vector (facoltativo)

        In inferenza usiamo solo yaw_gaze e pitch_gaze (weighted softmax → angoli).
        """

        def __init__(self, num_bins: int = 90):
            super().__init__()
            base = models.resnet50(weights=None)
            self.conv1 = base.conv1
            self.bn1 = base.bn1
            self.relu = base.relu
            self.maxpool = base.maxpool
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.avgpool = base.avgpool
            # Head names che corrispondono esattamente alle chiavi del checkpoint
            self.fc_yaw_gaze = nn.Linear(2048, num_bins)
            self.fc_pitch_gaze = nn.Linear(2048, num_bins)
            self.fc_finetune = nn.Linear(2051, 3)  # 3D gaze vector (opzionale)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            # Restituisce solo yaw e pitch bins (per la conversione ONNX)
            return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)

    return L2CS(num_bins=90)


def convert_to_onnx(use_fp16: bool = True):
    """Carica i pesi .pkl e converte in ONNX."""
    try:
        import torch
    except ImportError:
        print("[ERRORE] torch non installato. Eseguire: pip install torch")
        sys.exit(1)

    print("[L2CS] Caricamento architettura e pesi...")
    model = build_l2cs_model()

    # I modelli PyTorch sono salvati con torch.save(), non pickle.dump().
    # weights_only=False necessario per caricare oggetti arbitrari (state_dict dict).
    state_dict = torch.load(
        str(_WEIGHTS_FILE),
        map_location="cpu",
        weights_only=False,
    )

    # Se il checkpoint wrappa lo state_dict dentro una chiave (es. 'model', 'state_dict')
    if isinstance(state_dict, dict):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in state_dict:
                state_dict = state_dict[key]
                break

    # DataParallel salva con prefisso 'module.'
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, _ = model.load_state_dict(cleaned, strict=False)
    if missing:
        sample = missing[:3]
        print(f"[WARN] Chiavi mancanti ({len(missing)}): {sample}...")

    model.eval()
    dummy = torch.zeros(1, 3, 224, 224)

    print(f"[L2CS] Esportazione ONNX -> {OUTPUT_PATH}")
    # Usa l'API legacy (dynamo=False) per evitare dipendenza da onnxscript.
    # Se la versione torch e' >= 2.5 il parametro dynamo va passato esplicitamente.
    export_kwargs = dict(
        input_names=["input"],
        output_names=["yaw_logits", "pitch_logits"],
        dynamic_axes={"input": {0: "batch_size"}},
        export_params=True,
        do_constant_folding=True,
        opset_version=14,  # opset 14 e' sicuro su tutte le versioni torch >= 1.11
    )
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version >= (2, 5):
        # PyTorch >= 2.5 usa dynamo=True per default; forziamo il path legacy
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, dummy, str(OUTPUT_PATH), **export_kwargs)
    mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"[L2CS] ONNX FP32 salvato: {mb:.1f} MB")

    if use_fp16:
        _quantize_fp16()
    else:
        print("[L2CS] Skip FP16 (--no-fp16).")


def _quantize_fp16():
    """Converte ONNX FP32 -> FP16 (circa -50% dimensioni)."""
    for _try in ("onnxconverter", "onnxmltools"):
        try:
            import onnx

            if _try == "onnxconverter":
                from onnxconverter_common import float16

                print("[L2CS] Quantizzazione FP16 (onnxconverter-common)...")
                m = onnx.load(str(OUTPUT_PATH))
                m16 = float16.convert_float_to_float16(m, keep_io_types=True)
            else:
                from onnxmltools.utils.float16_converter import convert_float_to_float16

                print("[L2CS] Quantizzazione FP16 (onnxmltools)...")
                m = onnx.load(str(OUTPUT_PATH))
                m16 = convert_float_to_float16(m)
            onnx.save(m16, str(OUTPUT_PATH))
            print(f"[L2CS] Modello FP16: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")
            return
        except ImportError:
            continue

    print("[WARN] FP16 skip: installa onnxconverter-common o onnxmltools")
    print("       pip install onnxconverter-common")


def main():
    parser = argparse.ArgumentParser(
        description="Download + conversione L2CS-Net per GazeControl Enterprise"
    )
    parser.add_argument(
        "--no-fp16", action="store_true", help="Mantieni FP32 (~100 MB invece di ~50 MB)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Salta download (usa .pkl gia' presente in models/)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        ok = download_weights()
        if not ok:
            sys.exit(1)

    if not _WEIGHTS_FILE.exists():
        print(f"\n[ERRORE] File pesi non trovato: {_WEIGHTS_FILE}")
        print(_MANUAL_MSG.format(weights_path=_WEIGHTS_FILE))
        sys.exit(1)

    convert_to_onnx(use_fp16=not args.no_fp16)

    print()
    print("=" * 60)
    print("  [OK] L2CS-Net ONNX pronto!")
    print(f"       {OUTPUT_PATH}")
    print("=" * 60)
    print()
    print("  Prossimi passi:")
    print("  1. Calibra (se non l'hai gia' fatto):")
    print("     python gazecontrol/main.py --calibrate --profile default")
    print()
    print("  2. Avvia GazeControl (L2CS si attiva automaticamente):")
    print("     python gazecontrol/main.py --profile default")
    print()
    print("  3. Benchmark accuratezza:")
    print("     python tools/benchmark_gaze.py --profile default")


if __name__ == "__main__":
    main()
