"""BC 模型评估 — 帧堆叠版。

用法:
    python eval_stacked.py [checkpoint_path]
"""

import sys
import json
import random
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.bc_model import BCModel, NUM_BUTTONS, NUM_AXES
from preprocess.stacked_dataset import collect_sequences, StackedDataset
from preprocess.dataset import BUTTON_NAMES, AXIS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best_stacked.pt"
BATCH_SIZE = 64
MAX_SAMPLES = 5000
VAL_SPLIT = 0.2
NUM_WORKERS = 0


def evaluate():
    if len(sys.argv) > 1:
        ckpt_path = Path(sys.argv[1])
    else:
        ckpt_path = CHECKPOINT

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {ckpt_path}")

    if not ckpt_path.exists():
        print("Checkpoint not found!")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    seq_len = ckpt.get("seq_len", 1)

    # ── 模型 ──────────────────────────────────────────
    model = BCModel(num_frames=seq_len).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f}, seq_len={seq_len}")

    # ── 数据 ──────────────────────────────────────────
    all_seqs = collect_sequences(seq_len=seq_len, step=2, max_samples=MAX_SAMPLES)
    if len(all_seqs) == 0:
        all_seqs = collect_sequences(seq_len=seq_len, step=1, max_samples=MAX_SAMPLES)
    if len(all_seqs) == 0:
        print("No data found!")
        return

    random.shuffle(all_seqs)
    n_val = max(1, int(len(all_seqs) * VAL_SPLIT))
    val_seqs = all_seqs[:n_val]
    val_ds = StackedDataset(val_seqs, augment=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Eval sequences: {len(val_ds)}")

    # ── 推理 ──────────────────────────────────────────
    all_btn_true = []
    all_btn_pred = []
    all_axis_true = []
    all_axis_pred = []

    with torch.no_grad():
        for img, btns, axes in val_loader:
            img = img.to(DEVICE)
            btn_logits, axes_pred = model(img)
            all_btn_true.append(btns.numpy())
            all_btn_pred.append(torch.sigmoid(btn_logits).cpu().numpy())
            all_axis_true.append(axes.numpy())
            all_axis_pred.append(axes_pred.cpu().numpy())

    btn_true = np.concatenate(all_btn_true)
    btn_pred_prob = np.concatenate(all_btn_pred)
    axis_true = np.concatenate(all_axis_true)
    axis_pred = np.concatenate(all_axis_pred)
    btn_pred_bin = (btn_pred_prob >= 0.5).astype(np.float32)

    # ── 按钮 ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Button Metrics (per-key)")
    print("=" * 65)
    print(f"{'Key':<10} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'Pos%':>6}")
    print("-" * 50)

    for i, name in enumerate(BUTTON_NAMES):
        y_true = btn_true[:, i]
        y_pred = btn_pred_bin[:, i]
        if y_true.sum() > 0:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            pos_ratio = y_true.mean() * 100
            print(f"{name:<10} {acc:6.3f} {prec:6.3f} {rec:6.3f} {f1:6.3f}  {pos_ratio:5.1f}%")

    macro_acc = accuracy_score(btn_true.flatten(), btn_pred_bin.flatten())
    print(f"\n  Overall accuracy: {macro_acc:.4f}")

    # ── 摇杆 ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Axis Metrics (per-axis)")
    print("=" * 65)
    print(f"{'Axis':<10} {'MSE':>8} {'MAE':>8} {'Std_true':>8} {'Std_pred':>8}")
    print("-" * 50)

    for i, name in enumerate(AXIS_NAMES):
        mse = np.mean((axis_true[:, i] - axis_pred[:, i]) ** 2)
        mae = np.mean(np.abs(axis_true[:, i] - axis_pred[:, i]))
        std_t = np.std(axis_true[:, i])
        std_p = np.std(axis_pred[:, i])
        print(f"{name:<10} {mse:8.4f} {mae:8.4f} {std_t:8.4f} {std_p:8.4f}")

    overall_axis_mse = np.mean((axis_true - axis_pred) ** 2)
    overall_axis_mae = np.mean(np.abs(axis_true - axis_pred))
    print(f"\n  Overall axis MSE: {overall_axis_mse:.4f}")
    print(f"  Overall axis MAE: {overall_axis_mae:.4f}")


if __name__ == "__main__":
    evaluate()
