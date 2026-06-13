"""预处理数据集：从 preprocessed/ 目录加载已缩放的帧 + 标签。

比原始数据集快很多 — 不需要实时 resize，JPEG 文件也更小。
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


def collect_preprocessed(root: Path = PREPROCESSED_ROOT, max_samples: int = 0):
    """遍历预处理目录，返回 [(img_path, buttons, axes), ...] 列表。

    优先使用 labels_clean.json（清洗后），fallback 到 labels.json。
    """
    samples = []
    sessions = sorted(root.glob("session_*"))
    for sess in sessions:
        # 优先清洗后标签
        labels_path = sess / "labels_clean.json"
        if not labels_path.exists():
            labels_path = sess / "labels.json"
        frames_dir = sess / "frames"
        if not labels_path.exists() or not frames_dir.exists():
            continue

        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for fid, btns, axes in zip(data["frame_ids"], data["buttons"], data["axes"]):
            img_path = str(frames_dir / f"{fid:06d}.jpg")
            samples.append((img_path, btns, axes))

            if max_samples and len(samples) >= max_samples:
                return samples

    return samples


class PreprocessedDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
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
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, buttons, axes = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(buttons, dtype=torch.float32), torch.tensor(axes, dtype=torch.float32)
