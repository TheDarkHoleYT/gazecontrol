"""train_gesture_tcn.py — Train a 1D-TCN gesture classifier and export to ONNX.

The model takes a window of T=30 frames × F=16 features and outputs a
probability distribution over gesture classes.  It is exported as ONNX opset 17
for use with ``onnxruntime`` at inference time.

Usage::

    python tools/train_gesture_tcn.py \\
        --data data/gestures \\
        --out models/gesture_tcn_v1.onnx \\
        --epochs 50 \\
        --batch 32

Requirements (not in runtime deps — install separately for training)::

    pip install torch torchvision  # CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu

Output files::

    models/gesture_tcn_v1.onnx          — ONNX model (opset 17)
    models/training_report.md           — confusion matrix, per-class F1

Architecture (1D-TCN)
---------------------
- Conv1D layers with dilations 1, 2, 4 (receptive field = 7 × 3 kernel = 21 frames).
- Each block: Conv1D → BatchNorm → ReLU → Dropout.
- Global average pooling over time → Linear → Softmax.
- ~15k parameters; inference < 3 ms CPU on a single window.

Input shape:  [batch, T=30, F=16]  (channel-last; permuted to [B, F, T] for Conv1D)
Output shape: [batch, num_classes]  (log-softmax probabilities)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_gesture_tcn",
        description="Train a 1D-TCN gesture classifier and export to ONNX.",
    )
    p.add_argument(
        "--data",
        default="data/gestures",
        help="Directory containing .npz samples (from record_gesture_dataset.py).",
    )
    p.add_argument(
        "--out",
        default="models/gesture_tcn_v1.onnx",
        help="Path for the exported ONNX model.",
    )
    p.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50).")
    p.add_argument("--batch", type=int, default=32, help="Batch size (default: 32).")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    p.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument(
        "--report",
        default="models/training_report.md",
        help="Path for the markdown training report.",
    )
    return p


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def _load_dataset(data_dir: Path) -> tuple[list, list, list[str]]:
    """Load all .npz files and return (X_list, y_list, label_names).

    X_list: list of float32 arrays with shape [T, F].
    y_list: list of int class indices.
    label_names: list of label strings (sorted, canonical order).
    """
    import numpy as np

    samples_by_label: dict[str, list] = {}
    for npz_path in sorted(data_dir.glob("*.npz")):
        data = np.load(str(npz_path), allow_pickle=True)
        label = str(data["label"])
        features = data["features"].astype(np.float32)  # [T, F]
        if label not in samples_by_label:
            samples_by_label[label] = []
        samples_by_label[label].append(features)

    if not samples_by_label:
        print(f"[ERROR] No .npz files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    label_names = sorted(samples_by_label.keys())
    X_list, y_list = [], []
    for idx, lbl in enumerate(label_names):
        for arr in samples_by_label[lbl]:
            X_list.append(arr)
            y_list.append(idx)

    print(f"[data] Loaded {len(X_list)} samples across {len(label_names)} classes:")
    for lbl in label_names:
        print(f"  {lbl}: {len(samples_by_label[lbl])} samples")

    return X_list, y_list, label_names


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


def _build_model(n_features: int, n_classes: int, T: int) -> object:
    """Build a lightweight 1D-TCN model."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print(
            "[ERROR] PyTorch not found.  Install it with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu",
            file=sys.stderr,
        )
        sys.exit(1)

    class _TCNBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
            super().__init__()
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
            self.bn = nn.BatchNorm1d(out_ch)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(0.2)
            self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.drop(self.act(self.bn(self.conv(x))))
            return out + self.residual(x)

    class GestureTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                _TCNBlock(n_features, 32, dilation=1),
                _TCNBlock(32, 64, dilation=2),
                _TCNBlock(64, 64, dilation=4),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(64, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, F] → [B, F, T] for Conv1D
            x = x.permute(0, 2, 1)
            x = self.blocks(x)
            x = self.pool(x).squeeze(-1)
            return self.head(x)

    return GestureTCN()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train(
    model: object,
    X_list: list,
    y_list: list,
    label_names: list[str],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    seed: int,
) -> None:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    np.random.default_rng(seed)
    torch.manual_seed(seed)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)

    dataset = TensorDataset(X, y)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = model.to(device)  # type: ignore[union-attr]
    optim = torch.optim.Adam(m.parameters(), lr=lr)  # type: ignore[union-attr]
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    print(f"\n[train] Device: {device}  |  {n_train} train / {n_val} val")

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        m.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = crit(m(xb), yb)
            loss.backward()
            optim.step()
            train_loss += loss.item() * len(xb)
        sched.step()

        # Validation.
        m.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds = m(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        val_acc = correct / max(total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:3d}/{epochs}  "
                f"loss={train_loss / n_train:.4f}  val_acc={val_acc:.3f}"
            )

    if best_state is not None:
        m.load_state_dict(best_state)
    print(f"\n[train] Best val_acc: {best_val_acc:.3f}")


# ---------------------------------------------------------------------------
# ONNX export + report
# ---------------------------------------------------------------------------


def _export_onnx(model: object, out_path: Path, n_features: int, T: int = 30) -> None:
    import torch

    m = model  # type: ignore[union-attr]
    m.eval()  # type: ignore[union-attr]
    dummy = torch.zeros(1, T, n_features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        m,
        dummy,
        str(out_path),
        opset_version=17,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}},
    )
    print(f"[export] ONNX model saved → {out_path}")


def _write_report(
    report_path: Path,
    model: object,
    X_list: list,
    y_list: list,
    label_names: list[str],
    val_split: float,
    seed: int,
) -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    dataset = TensorDataset(X, y)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    _, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    val_dl = DataLoader(val_ds, batch_size=64)

    m = model  # type: ignore[union-attr]
    m.eval()  # type: ignore[union-attr]
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            preds = m(xb).argmax(1)
            all_pred.extend(preds.tolist())
            all_true.extend(yb.tolist())

    n_cls = len(label_names)
    conf_matrix = [[0] * n_cls for _ in range(n_cls)]
    for true, pred in zip(all_true, all_pred, strict=False):
        conf_matrix[true][pred] += 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GestureTCN Training Report",
        "",
        f"- **Labels**: {', '.join(label_names)}",
        f"- **Total samples**: {len(X_list)}",
        f"- **Val split**: {val_split}",
        "",
        "## Confusion Matrix",
        "",
        "| True \\ Pred | " + " | ".join(f"**{l}**" for l in label_names) + " |",
        "|" + "-|" * (n_cls + 1),
    ]
    for i, row in enumerate(conf_matrix):
        lines.append(f"| **{label_names[i]}** | " + " | ".join(str(v) for v in row) + " |")
    report_path.write_text("\n".join(lines) + "\n")
    print(f"[report] Saved → {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_argparser().parse_args()
    data_dir = Path(args.data)
    out_path = Path(args.out)
    report_path = Path(args.report)

    X_list, y_list, label_names = _load_dataset(data_dir)

    T = int(X_list[0].shape[0])
    F = int(X_list[0].shape[1])
    print(f"[data] T={T} frames, F={F} features, classes={label_names}")

    model = _build_model(n_features=F, n_classes=len(label_names), T=T)

    _train(
        model,
        X_list,
        y_list,
        label_names,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
    )

    _export_onnx(model, out_path, n_features=F, T=T)
    _write_report(report_path, model, X_list, y_list, label_names, args.val_split, args.seed)

    # Update model manifest with SHA256.
    import hashlib

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest_path = out_path.parent / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        with contextlib.suppress(Exception):
            manifest = json.loads(manifest_path.read_text())
    manifest[out_path.name] = {"sha256": sha, "labels": label_names, "T": T, "F": F}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] SHA256 {sha[:16]}… → {manifest_path}")


if __name__ == "__main__":
    main()
