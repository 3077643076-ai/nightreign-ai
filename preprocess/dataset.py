"""数据集：从 session 目录加载 帧图片 + 手柄标签。

输出格式：
  image: [3, H, W] normalized tensor
  buttons: [17] float (0/1)
  axes: [6] float ([-1, 1])
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from recorder.config import DATA_ROOT

BUTTON_NAMES = (
    "A", "B", "X", "Y", "LB", "RB", "BACK", "START",
    "LS", "RS", "GUIDE", "LT", "RT",
    "DPAD_U", "DPAD_D", "DPAD_L", "DPAD_R",
)
AXIS_NAMES = ("LX", "LY", "RX", "RY", "DPAD_X", "DPAD_Y")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _normalize(btn_raw, axis_raw):
    """统一新旧两种数据格式 → 标准 17 键 + 6 轴。"""
    buttons = []
    for k in BUTTON_NAMES:
        if k in btn_raw:
            buttons.append(float(btn_raw[k]))
        elif k in axis_raw:
            # 旧格式：LT/RT 是连续轴，二值化
            buttons.append(1.0 if axis_raw[k] >= 0.5 else 0.0)
        elif k == "DPAD_U":
            buttons.append(1.0 if axis_raw.get("DPAD_Y", 0) > 0.5 else 0.0)
        elif k == "DPAD_D":
            buttons.append(1.0 if axis_raw.get("DPAD_Y", 0) < -0.5 else 0.0)
        elif k == "DPAD_L":
            buttons.append(1.0 if axis_raw.get("DPAD_X", 0) < -0.5 else 0.0)
        elif k == "DPAD_R":
            buttons.append(1.0 if axis_raw.get("DPAD_X", 0) > 0.5 else 0.0)
        else:
            buttons.append(0.0)

    axes = []
    for k in AXIS_NAMES:
        if k in axis_raw:
            axes.append(float(axis_raw[k]))
        else:
            axes.append(0.0)

    return buttons, axes


def collect_samples(data_root: Path = DATA_ROOT, max_samples: int = 0, step: int = 1):
    """遍历所有 session，返回 [(frame_path, buttons, axes), ...] 列表。

    max_samples: 最大样本数，0=不限制。step: 隔 N 帧取 1 帧（降采样）。
    不做逐帧 exists() 检查（Windows 上太慢），文件缺失在 getitem 时处理。
    """
    samples = []
    sessions = sorted(data_root.glob("session_*"))
    for sess in sessions:
        inputs_path = sess / "inputs.jsonl"
        frames_dir = sess / "frames"
        if not inputs_path.exists() or not frames_dir.exists():
            continue

        sess_samples = []
        with open(inputs_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                if step > 1 and i % step != 0:
                    continue
                d = json.loads(line)
                frame_id = d["frame"]
                img_path = str(frames_dir / f"{frame_id:06d}.jpg")
                buttons, axes = _normalize(d["buttons"], d["axes"])
                sess_samples.append((img_path, buttons, axes))

        # 逐步合并，避免一次性分配大 list 导致 Python 3.13 崩溃
        if samples:
            samples.extend(sess_samples)
        else:
            samples = sess_samples

        if max_samples and len(samples) >= max_samples:
            return samples[:max_samples]

    return samples


class GamepadDataset(Dataset):
    def __init__(self, samples, img_size=224, augment=False):
        self.samples = samples
        tf = [transforms.Resize((img_size, img_size))]
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
