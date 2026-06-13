"""BC 训练入口：画面 → 手柄按键 + 摇杆。

用法：
    python -m train.train
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.bc_model import BCModel, NUM_BUTTONS, NUM_AXES
from preprocess.preprocessed_dataset import collect_preprocessed, PreprocessedDataset
from preprocess.dataset import BUTTON_NAMES, AXIS_NAMES

# ── 训练配置 ──────────────────────────────────────────────
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
MAX_SAMPLES = 30000  # 3万帧，每epoch约5分钟
SAMPLE_STEP = 1       # 隔帧采样，>1 可降采样
VAL_SPLIT = 0.1
NUM_WORKERS = 0  # Windows DataLoader 多进程容易崩，先单进程
GRADIENT_ACCUM_STEPS = 1
AMP_ENABLED = True
SAVE_EVERY = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def compute_pos_weight(samples):
    """从样本统计按钮正负比例，用于 BCE pos_weight。"""
    total = len(samples)
    if total == 0:
        return torch.ones(NUM_BUTTONS)

    pos = torch.zeros(NUM_BUTTONS)
    for _, btns, _ in samples:
        pos += torch.tensor(btns)
    neg = total - pos
    # pos_weight = neg / pos，避免除零
    pos_weight = neg.clone()
    pos_weight[pos > 0] = neg[pos > 0] / pos[pos > 0]
    pos_weight[pos == 0] = 0.0
    return pos_weight


def save_checkpoint(model, optimizer, scaler, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if AMP_ENABLED else None,
        "loss": loss,
    }, path)


def train(resume_from=None):
    # ── 数据 ──────────────────────────────────────────────
    print("Loading samples...")
    all_samples = collect_preprocessed(max_samples=MAX_SAMPLES)
    print(f"Total frames: {len(all_samples)}")

    if len(all_samples) == 0:
        print("No training data found!")
        return

    # 按比例切分
    random.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * VAL_SPLIT))
    train_samples = all_samples[n_val:]
    val_samples = all_samples[:n_val]
    n_train = len(train_samples)

    train_ds = PreprocessedDataset(train_samples, augment=True)
    val_ds = PreprocessedDataset(val_samples, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── 模型 ──────────────────────────────────────────────
    model = BCModel().to(DEVICE)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"Device: {DEVICE}")

    # ── 损失 ──────────────────────────────────────────────
    pos_weight = compute_pos_weight(all_samples).to(DEVICE)
    print(f"Button pos_weight: {[f'{w:.1f}' for w in pos_weight.tolist()]}")

    btn_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    axis_criterion = nn.SmoothL1Loss()

    # ── 优化器 ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda") if AMP_ENABLED else None

    # ── 训练循环 ──────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    log_lines = []

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────
        model.train()
        train_btn_loss = 0.0
        train_axis_loss = 0.0
        optimizer.zero_grad()

        for step, (img, btns, axes) in enumerate(train_loader):
            img, btns, axes = img.to(DEVICE), btns.to(DEVICE), axes.to(DEVICE)

            with autocast("cuda", enabled=AMP_ENABLED):
                btn_logits, axes_pred = model(img)
                l_btn = btn_criterion(btn_logits, btns)
                l_axis = axis_criterion(axes_pred, axes)
                loss = l_btn + l_axis
                loss = loss / GRADIENT_ACCUM_STEPS

            if AMP_ENABLED:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % GRADIENT_ACCUM_STEPS == 0:
                if AMP_ENABLED:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            train_btn_loss += l_btn.item()
            train_axis_loss += l_axis.item()

        train_btn_loss /= len(train_loader)
        train_axis_loss /= len(train_loader)

        # ── Val ───────────────────────────────────────────
        model.eval()
        val_btn_loss = 0.0
        val_axis_loss = 0.0

        with torch.no_grad():
            for img, btns, axes in val_loader:
                img, btns, axes = img.to(DEVICE), btns.to(DEVICE), axes.to(DEVICE)
                btn_logits, axes_pred = model(img)
                val_btn_loss += btn_criterion(btn_logits, btns).item()
                val_axis_loss += axis_criterion(axes_pred, axes).item()

        val_btn_loss /= len(val_loader)
        val_axis_loss /= len(val_loader)
        val_total = val_btn_loss + val_axis_loss

        # ── Log ───────────────────────────────────────────
        msg = (f"Epoch {epoch:3d}/{EPOCHS} | "
               f"Train btn={train_btn_loss:.4f} axis={train_axis_loss:.4f} | "
               f"Val btn={val_btn_loss:.4f} axis={val_axis_loss:.4f}")
        print(msg)
        log_lines.append(msg)

        scheduler.step()

        # ── Save ──────────────────────────────────────────
        if val_total < best_val_loss:
            best_val_loss = val_total
            save_checkpoint(model, optimizer, scaler, epoch, val_total,
                            OUTPUT_DIR / "best.pt")
            print(f"  → best model saved (val_loss={best_val_loss:.4f})")

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, scaler, epoch, val_total,
                            OUTPUT_DIR / f"epoch_{epoch:03d}.pt")

    # ── 最终保存 ──────────────────────────────────────────
    save_checkpoint(model, optimizer, scaler, EPOCHS, val_total,
                    OUTPUT_DIR / "last.pt")

    # 训练日志
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"train_log_{ts}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "batch_size": BATCH_SIZE, "img_size": IMG_SIZE,
                "lr": LR, "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS,
                "train_frames": n_train, "val_frames": n_val,
            },
            "log": log_lines,
        }, f, indent=2, ensure_ascii=False)

    print(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
