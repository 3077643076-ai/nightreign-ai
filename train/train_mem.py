"""BC 训练 — 内存版：所有帧预加载到 RAM，零磁盘 I/O。

启动时约 30 秒顺序读取全部 JPEG 到内存（~3-4GB），
之后训练全程无磁盘访问，随机打乱不会导致 HDD 颠簸。

用法：
    python -m train.train_mem
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.bc_model import BCModel, NUM_BUTTONS, NUM_AXES
from preprocess.memory_dataset import (
    load_all_sessions, collect_in_memory, InMemoryDataset,
)
from preprocess.dataset import BUTTON_NAMES, AXIS_NAMES

BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
SEQ_LEN = 4
MAX_SAMPLES = 0  # 0 = 全部
STEP = 2         # 滑动窗口步长（2 可减半序列数）
VAL_SPLIT = 0.1
NUM_WORKERS = 2  # 内存解码可以用多线程了
AMP_ENABLED = True
SAVE_EVERY = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def compute_pos_weight(sequences):
    total = len(sequences)
    if total == 0:
        return torch.ones(NUM_BUTTONS)
    pos = torch.zeros(NUM_BUTTONS)
    for _, _, btns, _ in sequences:
        pos += torch.tensor(btns)
    neg = total - pos
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
        "seq_len": SEQ_LEN,
    }, path)


def train():
    print(f"Seq len: {SEQ_LEN} | Device: {DEVICE}")
    print(f"Batch: {BATCH_SIZE} | LR: {LR} | Epochs: {EPOCHS}")

    # ── 1. 预加载所有 session 到内存 ──────────────────
    print("\nLoading sessions into RAM...")
    import time
    t0 = time.time()
    all_buffers = load_all_sessions()
    elapsed = time.time() - t0
    total_frames = sum(len(b) for b in all_buffers.values())
    print(f"  {len(all_buffers)} sessions, {total_frames // 1000}k frames "
          f"loaded in {elapsed:.1f}s")

    # ── 2. 收集序列 ──────────────────────────────────
    print("\nCollecting sequences...")
    all_seqs = collect_in_memory(all_buffers, seq_len=SEQ_LEN, step=STEP,
                                 max_samples=MAX_SAMPLES)
    print(f"  {len(all_seqs)} sequences")

    if len(all_seqs) == 0:
        print("No training data!")
        return

    random.shuffle(all_seqs)
    n_val = max(1, int(len(all_seqs) * VAL_SPLIT))
    train_seqs = all_seqs[n_val:]
    val_seqs = all_seqs[:n_val]
    n_train = len(train_seqs)
    n_val = len(val_seqs)
    print(f"  Train: {n_train} | Val: {n_val}")

    # ── 3. 创建 Dataset / DataLoader ─────────────────
    train_ds = InMemoryDataset(train_seqs, all_buffers, augment=True)
    val_ds = InMemoryDataset(val_seqs, all_buffers, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── 4. 模型 ──────────────────────────────────────
    model = BCModel(num_frames=SEQ_LEN).to(DEVICE)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    pos_weight = compute_pos_weight(all_seqs).to(DEVICE)
    btn_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    axis_criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda") if AMP_ENABLED else None

    # ── 5. 训练循环 ──────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    log_lines = []

    for epoch in range(1, EPOCHS + 1):
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

            if AMP_ENABLED:
                scaler.scale(loss).backward()
            else:
                loss.backward()

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

        # Val
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

        msg = (f"Epoch {epoch:3d}/{EPOCHS} | "
               f"Train btn={train_btn_loss:.4f} axis={train_axis_loss:.4f} | "
               f"Val btn={val_btn_loss:.4f} axis={val_axis_loss:.4f}")
        print(msg)
        log_lines.append(msg)

        scheduler.step()

        if val_total < best_val_loss:
            best_val_loss = val_total
            save_checkpoint(model, optimizer, scaler, epoch, val_total,
                            OUTPUT_DIR / "best_stacked.pt")
            print(f"  -> best model saved (val_loss={best_val_loss:.4f})")

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, scaler, epoch, val_total,
                            OUTPUT_DIR / f"epoch_stacked_{epoch:03d}.pt")

    # 最终保存
    save_checkpoint(model, optimizer, scaler, EPOCHS, val_total,
                    OUTPUT_DIR / "last_stacked.pt")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"train_mem_log_{ts}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS,
                "seq_len": SEQ_LEN, "step": STEP,
                "train_sequences": n_train, "val_sequences": n_val,
            },
            "log": log_lines,
        }, f, indent=2, ensure_ascii=False)

    print(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
