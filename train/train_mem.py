"""BC 训练 — 内存版：所有帧预加载到 RAM，零磁盘 I/O。
Focal Loss 解决按键类不平衡 + 早停防过拟合。

用法：
    python -m train.train_mem              # 全部数据
    python -m train.train_mem --mode combat  # 只战斗帧
    python -m train.train_mem --mode explore # 只跑图帧
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
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
EPOCHS = 20                # 上限，早停会提前结束
SEQ_LEN = 4
MAX_SAMPLES = 8000   # 懒加载下 HDD 随机读慢，缩小数据集
STEP = 3
VAL_SPLIT = 0.1
NUM_WORKERS = 0
AMP_ENABLED = True
SAVE_EVERY = 5

# 早停配置
PATIENCE = 5               # 连续 PATIENCE 个 epoch val loss 不降就停
OVERFIT_PATIENCE = 3       # 连续 OVERFIT_PATIENCE 个 epoch 过拟合（train↓ val↑）就停
FOCAL_GAMMA = 2.0          # Focal Loss 的 gamma 参数，越大越"欺软怕硬"

# 每个按钮的正样本权重：稀有但重要的按钮给更高权重
# 顺序对应 BUTTON_NAMES: A,B,X,Y,LB,RB,BACK,START,LS,RS,GUIDE,LT,RT,DPAD_U/D/L/R
# pos_weight > 1 → 漏按的惩罚更重，逼模型更积极按
_BTN_POS_WEIGHT_LIST = [
    1.0,   # A  - 闪避（常见）
    1.0,   # B  - 后撤（常见）
    1.0,   # X  - 喝药（常见）
    1.0,   # Y  - 交互（中等）
    1.0,   # LB - 防御（常见）
    1.0,   # RB - 轻击（常见）
    1.0,   # BACK - 菜单（极少，不重要）
    1.0,   # START - 菜单（极少，不重要）
    2.0,   # LS - 疾跑（不太常见但有用）
    8.0,   # RS - 锁定!!! 录的时候只按一次但极其重要!!
    1.0,   # GUIDE - 从不用
    1.0,   # LT - 战技（中等）
    1.0,   # RT - 重击（中等）
    1.0,   # DPAD_U（换道具，中等）
    1.0,   # DPAD_D（换道具，中等）
    1.0,   # DPAD_L（换左手，较少）
    1.0,   # DPAD_R（换右手，较少）
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Loss 二分类版：自动降低"简单样本"的权重，让模型专注学难的。

    公式: FL(p_t) = -(1 - p_t)^γ * log(p_t)
    其中 p_t = p if y=1 else 1-p，即模型对「正确答案」的信心。
    当模型信心很高时 (1-p_t) 接近 0，样本被大幅降权。
    """

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight  # 正样本权重，和 BCE 用法一致

    def forward(self, logits, targets):
        # 标准 BCE loss (不减缩)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        # 模型对正确答案的信心 p_t
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        # focal 权重: 信心越高，权重越低
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        # pos_weight: 叠加正样本加权
        if self.pos_weight is not None:
            alpha = targets * self.pos_weight + (1 - targets) * 1.0
            loss = loss * alpha
        return loss.mean()


def save_checkpoint(model, optimizer, scaler, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if AMP_ENABLED else None,
        "loss": loss,
        "seq_len": SEQ_LEN,
    }, path)


def train(mode=None):
    """训练 BC 模型。

    mode: None=全部数据, "combat"=只战斗帧, "explore"=只探索帧
    """
    mode_str = f" [{mode}]" if mode else " [all]"
    print(f"Seq len: {SEQ_LEN} | Device: {DEVICE} | Mode:{mode_str}")
    print(f"Batch: {BATCH_SIZE} | LR: {LR} | Epochs max: {EPOCHS}")
    print(f"Focal γ={FOCAL_GAMMA} | Patience: {PATIENCE} | Overfit patience: {OVERFIT_PATIENCE}")

    # 根据模式调整 max_samples
    max_samp = MAX_SAMPLES
    if mode == "combat":
        max_samp = 4000  # 战斗帧约 90k，4k 序列足够
    elif mode == "explore":
        max_samp = MAX_SAMPLES  # 跑图帧多，限采样

    # 输出文件名
    suffix = f"_{mode}" if mode else "_focal"
    best_path = OUTPUT_DIR / f"best{suffix}.pt"
    last_path = OUTPUT_DIR / f"last{suffix}.pt"
    epoch_prefix = f"epoch_{mode}_" if mode else "epoch_focal_"

    # ── 1. 预加载所有 session 到内存 ──────────────────
    print("\nLoading sessions into RAM...")
    import time
    t0 = time.time()
    all_buffers = load_all_sessions(min_frames=100, max_total_frames=400000)
    elapsed = time.time() - t0
    total_frames = sum(len(b) for b in all_buffers.values())
    print(f"  {len(all_buffers)} sessions, {total_frames // 1000}k frames "
          f"loaded in {elapsed:.1f}s")

    # ── 2. 收集序列 ──────────────────────────────────
    print(f"\nCollecting sequences (mode={mode or 'all'})...")
    all_seqs = collect_in_memory(all_buffers, seq_len=SEQ_LEN, step=STEP,
                                 max_samples=max_samp, mode=mode)
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

    # Focal Loss 用于按钮 + pos_weight 让模型更重视 RS 等稀有按键
    pos_w = torch.tensor(_BTN_POS_WEIGHT_LIST, device=DEVICE)
    btn_criterion = FocalBCEWithLogitsLoss(gamma=FOCAL_GAMMA, pos_weight=pos_w)
    axis_criterion = nn.SmoothL1Loss()
    print(f"  BTN pos_weight: RS={_BTN_POS_WEIGHT_LIST[9]:.0f}x LS={_BTN_POS_WEIGHT_LIST[8]:.0f}x")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda") if AMP_ENABLED else None

    # ── 5. 训练循环 ──────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    overfit_counter = 0
    prev_train_btn = None
    prev_val_btn = None
    log_lines = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # — 训练 —
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
        train_total = train_btn_loss + train_axis_loss

        # — 验证 —
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

        elapsed = time.time() - epoch_start

        # — 诊断 —
        overfit_flag = ""
        if prev_train_btn is not None:
            train_btn_down = train_btn_loss < prev_train_btn - 1e-4  # train 下降
            val_btn_up = val_btn_loss > prev_val_btn + 1e-4            # val 上升
            if train_btn_down and val_btn_up:
                overfit_counter += 1
                overfit_flag = f" !!OVERFIT x{overfit_counter}"
            else:
                overfit_counter = 0

        msg = (f"Epoch {epoch:3d}/{EPOCHS} | "
               f"Train btn={train_btn_loss:.4f} axis={train_axis_loss:.4f} | "
               f"Val btn={val_btn_loss:.4f} axis={val_axis_loss:.4f} | "
               f"{elapsed:.0f}s{overfit_flag}")
        print(msg)
        log_lines.append(msg)

        prev_train_btn = train_btn_loss
        prev_val_btn = val_btn_loss

        scheduler.step()

        # — 保存最佳 —
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, optimizer, scaler, epoch, val_total, best_path)
            print(f"  -> best model saved (epoch {epoch}, val={val_total:.4f})")
        else:
            patience_counter += 1

        # 定期保存
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, scaler, epoch, val_total,
                            OUTPUT_DIR / f"{epoch_prefix}{epoch:03d}.pt")

        # — 早停检查 —
        if overfit_counter >= OVERFIT_PATIENCE:
            print(f"\n  连续 {OVERFIT_PATIENCE} 个 epoch 过拟合 (train↓ val↑)，提前停止！")
            break
        if patience_counter >= PATIENCE:
            print(f"\n  Val loss {PATIENCE} 个 epoch 没有改善，提前停止！")
            break

    # 最终保存
    save_checkpoint(model, optimizer, scaler, epoch, val_total, last_path)

    log_prefix = f"train_{mode or 'focal'}_log_"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"{log_prefix}{ts}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "mode": mode or "all",
                "batch_size": BATCH_SIZE, "lr": LR, "epochs_max": EPOCHS,
                "epochs_run": epoch, "seq_len": SEQ_LEN, "step": STEP,
                "train_sequences": n_train, "val_sequences": n_val,
                "focal_gamma": FOCAL_GAMMA, "patience": PATIENCE,
                "overfit_patience": OVERFIT_PATIENCE,
            },
            "best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "log": log_lines,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Best: epoch {best_epoch}, val={best_val_loss:.4f}")
    print(f"Checkpoints: {best_path.name}, {last_path.name}")


if __name__ == "__main__":
    # 解析命令行参数
    mode = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
    if mode not in (None, "combat", "explore"):
        print("用法: python -m train.train_mem [--mode combat|explore]")
        sys.exit(1)
    train(mode=mode)
