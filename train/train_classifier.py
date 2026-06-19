"""训练画面分类器：判断当前画面是 探索(0) 还是 战斗(1)。

输入：单帧画面 (3, 224, 224)
输出：二分类 logits (explore, combat)

用法：
    python -m train.train_classifier
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess.memory_dataset import load_all_sessions, PREPROCESSED_ROOT

# === 配置 ===
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
NUM_WORKERS = 0
AMP_ENABLED = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

# 采样：战斗/探索各取多少帧（平衡采样）
MAX_SAMPLES_PER_CLASS = 10000
IMG_SIZE = 224


class ClassifierDataset(Dataset):
    """单帧 + combat_labels 的数据集。"""

    def __init__(self, buffers, combat_labels_map, fids_list, labels_list):
        self.buffers = buffers
        self.cl_map = combat_labels_map  # sess_name -> np.array
        self.samples = []  # [(sess_name, fid, label)]

        # 平衡采样
        combat_samples = []
        explore_samples = []
        for sess_name, fids in fids_list.items():
            cl = self.cl_map.get(sess_name)
            if cl is None:
                continue
            for i, fid in enumerate(fids):
                if i < len(cl):
                    lbl = int(cl[i])
                    if lbl == 1:
                        combat_samples.append((sess_name, fid, lbl))
                    else:
                        explore_samples.append((sess_name, fid, lbl))

        # 各取 MAX_SAMPLES_PER_CLASS
        n_combat = min(len(combat_samples), MAX_SAMPLES_PER_CLASS)
        n_explore = min(len(explore_samples), MAX_SAMPLES_PER_CLASS)
        import random
        self.samples = (
            random.sample(combat_samples, n_combat) +
            random.sample(explore_samples, n_explore)
        )
        random.shuffle(self.samples)
        print(f"  Classifier dataset: {n_combat} combat + {n_explore} explore "
              f"= {len(self.samples)} total")

        # normalize (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sess_name, fid, lbl = self.samples[idx]
        buf = self.buffers[sess_name]
        img = buf.decode(fid)  # returns PIL Image
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img = (img - self.mean) / self.std
        return img, torch.tensor(lbl, dtype=torch.long)


class Classifier(nn.Module):
    """轻量二分类器：ResNet-18 backbone → binary output。"""

    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(512, 2)  # 2 分类

    def forward(self, x):
        return self.backbone(x)


def train():
    print(f"Device: {DEVICE} | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"Max samples per class: {MAX_SAMPLES_PER_CLASS}")

    # ── 1. 加载数据 ──
    print("\nLoading sessions...")
    buffers = load_all_sessions(max_total_frames=300000)

    # 加载所有 frame_ids 和 combat_labels
    fids_map = {}
    cl_map = {}
    for sess_name, buf in buffers.items():
        labels_path = PREPROCESSED_ROOT / sess_name / "labels_clean.json"
        if not labels_path.exists():
            labels_path = PREPROCESSED_ROOT / sess_name / "labels.json"
        if not labels_path.exists():
            continue
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fids_map[sess_name] = data["frame_ids"]

        cl_path = PREPROCESSED_ROOT / sess_name / "combat_labels.npy"
        if cl_path.exists():
            cl_map[sess_name] = np.load(cl_path)
            n_combat = cl_map[sess_name].sum()
            print(f"  {sess_name}: {len(data['frame_ids'])} frames, "
                  f"{int(n_combat)} combat ({n_combat/len(data['frame_ids'])*100:.1f}%)")

    # ── 2. 创建数据集 ──
    print("\nBuilding balanced dataset...")
    ds = ClassifierDataset(buffers, cl_map, fids_map, None)
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    print(f"  Train: {n_train} | Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # ── 3. 模型 ──
    model = Classifier().to(DEVICE)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda") if AMP_ENABLED else None

    # ── 4. 训练 ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    PATIENCE = 7

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        for img, lbl in train_loader:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                logits = model(img)
                loss = criterion(logits, lbl)

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

            train_loss += loss.item()
            train_correct += (logits.argmax(1) == lbl).sum().item()
            train_total += lbl.size(0)

        train_acc = train_correct / train_total

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for img, lbl in val_loader:
                img, lbl = img.to(DEVICE), lbl.to(DEVICE)
                logits = model(img)
                val_loss += criterion(logits, lbl).item()
                val_correct += (logits.argmax(1) == lbl).sum().item()
                val_total += lbl.size(0)

        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train loss={train_loss/len(train_loader):.4f} acc={train_acc:.3f} | "
              f"Val loss={val_loss/len(val_loader):.4f} acc={val_acc:.3f} | "
              f"{elapsed:.0f}s")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            path = OUTPUT_DIR / "best_classifier.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, path)
            print(f"  -> best saved (acc={val_acc:.4f})")
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc},
                       OUTPUT_DIR / f"epoch_classifier_{epoch:03d}.pt")

        if patience_counter >= PATIENCE:
            print(f"\n  连续 {PATIENCE} 个 epoch 没改善，提前停止！")
            break

    # 最终保存
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc},
               OUTPUT_DIR / "last_classifier.pt")

    print(f"\nDone. Best: epoch {best_epoch}, val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    train()
