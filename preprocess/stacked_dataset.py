"""帧堆叠数据集：连续 N 帧 → 最后一帧的标签。

单帧看不出运动方向和时机，堆叠 3-4 帧让模型感知动作。
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

PREPROCESSED_ROOT = Path(__file__).resolve().parent.parent / "preprocessed"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def collect_sequences(root=PREPROCESSED_ROOT, seq_len=4, step=1, max_samples=0):
    """遍历预处理目录，返回 [(frame_paths, buttons, axes), ...]。

    seq_len: 连续帧数
    step: 滑动窗口步长（>1 可降采样减少重叠）
    frame_paths 按时序排列，标签取最后一帧。
    优先使用 labels_clean.json。
    """
    sequences = []
    for sess in sorted(root.glob("session_*")):
        labels_path = sess / "labels_clean.json"
        if not labels_path.exists():
            labels_path = sess / "labels.json"
        frames_dir = sess / "frames"
        if not labels_path.exists() or not frames_dir.exists():
            continue

        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fids = data["frame_ids"]
        btns = data["buttons"]
        axes = data["axes"]
        n = len(fids)
        if n < seq_len:
            continue

        for i in range(0, n - seq_len + 1, step):
            paths = [str(frames_dir / f"{fids[j]:06d}.jpg") for j in range(i, i + seq_len)]
            sequences.append((paths, btns[i + seq_len - 1], axes[i + seq_len - 1]))

            if max_samples and len(sequences) >= max_samples:
                return sequences

    return sequences


class StackedDataset(Dataset):
    def __init__(self, sequences, augment=False):
        self.sequences = sequences
        tf = []
        if augment:
            tf += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        tf += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        self.transform = transforms.Compose(tf)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        paths, buttons, axes = self.sequences[idx]
        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            img = self.transform(img)
            frames.append(img)
        # [seq_len*3, H, W]
        stacked = torch.cat(frames, dim=0)
        return stacked, torch.tensor(buttons, dtype=torch.float32), torch.tensor(axes, dtype=torch.float32)
