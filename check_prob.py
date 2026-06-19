"""分析模型原始概率输出：在"人类按了"vs"没按"的帧上对比。"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from models.bc_model import BCModel, NUM_BUTTONS
from preprocess.memory_dataset import (
    load_all_sessions, collect_in_memory, InMemoryDataset,
)
from preprocess.dataset import BUTTON_NAMES
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = Path(__file__).resolve().parent / "checkpoints" / "best_focal.pt"

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CKPT}")

ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
seq_len = ckpt.get("seq_len", 4)
print(f"Epoch: {ckpt['epoch']}, loss: {ckpt['loss']:.4f}, seq_len: {seq_len}")

model = BCModel(num_frames=seq_len).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("\nLoading data...")
buffers = load_all_sessions()
seqs = collect_in_memory(buffers, seq_len=seq_len, step=3, max_samples=5000)

import random
random.shuffle(seqs)
ds = InMemoryDataset(seqs, buffers, augment=False)
loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

all_probs = []
all_labels = []
with torch.no_grad():
    for img, btns, axes in loader:
        img = img.to(DEVICE)
        logits, _ = model(img)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(btns.numpy())

probs = np.concatenate(all_probs)
labels = np.concatenate(all_labels)

print(f"\n{'='*70}")
print(f"  模型概率分析：按键按下 (label=1) vs 松开 (label=0) 的概率均值")
print(f"{'='*70}")
print(f"{'Key':<10} {'P(label=1)':>12} {'P(label=0)':>12} {'Diff':>8} {'Δ/σ':>8}")
print(f"{'':10} {'(真按了)':>12} {'(真没按)':>12}")
print(f"{'-'*55}")

for i, name in enumerate(BUTTON_NAMES[:11]):  # 前11个是实际按键
    pos_mask = labels[:, i] == 1
    neg_mask = labels[:, i] == 0

    prob_pos = probs[pos_mask, i].mean() if pos_mask.sum() > 0 else 0
    prob_neg = probs[neg_mask, i].mean() if neg_mask.sum() > 0 else 0
    std_pos = probs[pos_mask, i].std() if pos_mask.sum() > 1 else 0

    diff = prob_pos - prob_neg
    d_sigma = diff / std_pos if std_pos > 0 else 0

    marker = ""
    if diff > 0.1:
        marker = " <-- 真学到了！"
    elif diff > 0.03:
        marker = " <-- 有区分能力"
    elif diff > 0:
        marker = " (微弱)"
    else:
        marker = " (反向)"

    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    print(f"{name:<10} {prob_pos:12.4f} {prob_neg:12.4f} {diff:8.4f} {d_sigma:8.2f}{marker} "
          f"(n_pos={n_pos}, n_neg={n_neg})")

print(f"\nDiff = P(真按了) - P(真没按)，越大越好(・ω・)b")
