"""BC 模型评估：计算按钮/摇杆的各项指标。

用法:
    python eval_bc.py
"""

import sys
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.bc_model import BCModel, NUM_BUTTONS, NUM_AXES
from preprocess.preprocessed_dataset import collect_preprocessed, PreprocessedDataset
from preprocess.dataset import BUTTON_NAMES, AXIS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best.pt"
BATCH_SIZE = 64
MAX_SAMPLES = 5000  # 评估样本数，太多会慢
VAL_SPLIT = 0.2
NUM_WORKERS = 0


def evaluate():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")

    # ── 加载模型 ──────────────────────────────────────────
    model = BCModel().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f}")

    # ── 加载数据 ──────────────────────────────────────────
    all_samples = collect_preprocessed(max_samples=MAX_SAMPLES)
    if len(all_samples) == 0:
        print("No data found!")
        return

    random.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * VAL_SPLIT))
    val_samples = all_samples[:n_val]
    val_ds = PreprocessedDataset(val_samples, augment=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Eval samples: {len(val_ds)}")

    # ── 推理 ──────────────────────────────────────────────
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

    # ── 按钮指标 ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Button Metrics (per-key)")
    print("=" * 65)
    print(f"{'Key':<10} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'Pos%':>6}")
    print("-" * 50)

    total_pos = 0
    for i, name in enumerate(BUTTON_NAMES):
        y_true = btn_true[:, i]
        y_pred = btn_pred_bin[:, i]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pos_ratio = y_true.mean() * 100
        total_pos += y_true.sum()
        if y_true.sum() > 0:  # 只打印有正样本的键
            print(f"{name:<10} {acc:6.3f} {prec:6.3f} {rec:6.3f} {f1:6.3f}  {pos_ratio:5.1f}%")

    # 总体按钮准确率（宏平均 + 加权）
    macro_acc = accuracy_score(btn_true.flatten(), btn_pred_bin.flatten())
    print(f"\n  Overall accuracy: {macro_acc:.4f}")
    print(f"  Total button presses in eval set: {int(total_pos)}")

    # ── 摇杆指标 ──────────────────────────────────────────
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

    # ── 结果汇总 ──────────────────────────────────────────
    results = {
        "num_eval_samples": len(val_ds),
        "checkpoint": str(CHECKPOINT.name),
        "overall_button_accuracy": float(macro_acc),
        "overall_axis_mse": float(overall_axis_mse),
        "overall_axis_mae": float(overall_axis_mae),
        "per_button": {},
        "per_axis": {},
    }

    for i, name in enumerate(BUTTON_NAMES):
        y_true = btn_true[:, i]
        y_pred = btn_pred_bin[:, i]
        if y_true.sum() > 0:
            results["per_button"][name] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "positive_ratio": float(y_true.mean()),
            }

    for i, name in enumerate(AXIS_NAMES):
        results["per_axis"][name] = {
            "mse": float(np.mean((axis_true[:, i] - axis_pred[:, i]) ** 2)),
            "mae": float(np.mean(np.abs(axis_true[:, i] - axis_pred[:, i]))),
        }

    results_path = Path(__file__).resolve().parent / "checkpoints" / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    evaluate()
