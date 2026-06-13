"""内存数据集：所有帧的 JPEG 字节预加载到 RAM，训练时零磁盘 I/O。

支持两种加载方式：
- .pak 文件（快）：单个文件顺序读取，~10s/session
- 单个 JPEG 文件（慢）：HDD 上 60k 文件随机读取，可能几分钟

先运行 python -m preprocess.pack 打包，再训练。
"""

import json
import struct
from pathlib import Path
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

PREPROCESSED_ROOT = Path(__file__).resolve().parent.parent / "preprocessed"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SessionBuffer:
    """把一个 session 的所有帧 JPEG 字节读入 RAM。"""

    def __init__(self, session_dir: Path):
        self._bytes: dict[int, bytes] = {}
        pak_path = session_dir / "frames.pak"
        if pak_path.exists():
            self._load_pak(pak_path)
        else:
            self._load_files(session_dir / "frames")
        self._frame_ids = sorted(self._bytes.keys())

    def _load_pak(self, pak_path: Path):
        """从单个 .pak 文件顺序读取所有帧（快，HDD 友好）。"""
        data = pak_path.read_bytes()
        num_frames, index_offset = struct.unpack_from("<IQ", data, 0)

        # 解析索引
        idx_data = data[index_offset:index_offset + num_frames * 16]
        for i in range(num_frames):
            off = i * 16
            fid, frame_off, frame_len = struct.unpack_from("<IQI", idx_data, off)
            # JPEG 数据在 length prefix 之后
            jpg_start = frame_off + 4
            self._bytes[fid] = data[jpg_start:jpg_start + frame_len]

    def _load_files(self, frames_dir: Path):
        """逐个文件读取（慢，仅 fallback）。"""
        import warnings
        warnings.warn(f"No .pak found in {frames_dir.parent}, "
                      f"reading individual files (slow). "
                      f"Run: python -m preprocess.pack")
        for p in sorted(frames_dir.glob("*.jpg")):
            fid = int(p.stem)
            self._bytes[fid] = p.read_bytes()

    def __len__(self):
        return len(self._bytes)

    def decode(self, frame_id: int) -> Image.Image:
        return Image.open(BytesIO(self._bytes[frame_id])).convert("RGB")

    @property
    def frame_ids(self):
        return self._frame_ids


def load_all_sessions(root=PREPROCESSED_ROOT):
    """加载所有 session 到内存，返回 {session_name: SessionBuffer}。"""
    buffers = {}
    for sess_dir in sorted(root.glob("session_*")):
        has_frames = (sess_dir / "frames").exists()
        has_pak = (sess_dir / "frames.pak").exists()
        if not has_frames and not has_pak:
            continue
        print(f"  Loading {sess_dir.name}...", end=" ", flush=True)
        import time
        t0 = time.time()
        buf = SessionBuffer(sess_dir)
        t = time.time() - t0
        print(f"{len(buf)} frames in {t:.1f}s")
        buffers[sess_dir.name] = buf
    return buffers


def collect_in_memory(buffers: dict, seq_len=4, step=1, max_samples=0,
                      exclude_sessions: set = None):
    """从 SessionBuffer 字典收集帧堆叠序列。"""
    sequences = []
    for sess_name, buf in buffers.items():
        if exclude_sessions and sess_name in exclude_sessions:
            continue

        labels_path = PREPROCESSED_ROOT / sess_name / "labels_clean.json"
        if not labels_path.exists():
            labels_path = PREPROCESSED_ROOT / sess_name / "labels.json"
        if not labels_path.exists():
            continue

        with open(labels_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        fids = label_data["frame_ids"]
        btns = label_data["buttons"]
        axes = label_data["axes"]
        n = len(fids)
        if n < seq_len:
            continue

        for i in range(0, n - seq_len + 1, step):
            seq_fids = [fids[j] for j in range(i, i + seq_len)]
            if all(fid in buf._bytes for fid in seq_fids):
                sequences.append((sess_name, seq_fids,
                                  btns[i + seq_len - 1],
                                  axes[i + seq_len - 1]))
                if max_samples and len(sequences) >= max_samples:
                    return sequences

    return sequences


class InMemoryDataset(Dataset):
    """从内存中的 SessionBuffer 字典加载帧堆叠序列。零磁盘 I/O。"""

    def __init__(self, sequences, buffers: dict, augment=False):
        self.sequences = sequences
        self.buffers = buffers
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
        sess_name, fids, buttons, axes = self.sequences[idx]
        buf = self.buffers[sess_name]
        frames = []
        for fid in fids:
            img = buf.decode(fid)
            img = self.transform(img)
            frames.append(img)
        stacked = torch.cat(frames, dim=0)
        return (stacked,
                torch.tensor(buttons, dtype=torch.float32),
                torch.tensor(axes, dtype=torch.float32))
