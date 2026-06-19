"""混合数据集：大 session 用 pak 文件按需读取，小 session 预加载到内存。

有 .pak 的 session → 保持文件句柄打开，seek+read 按需读取（内存 ~MB）
无 .pak 的 session → 逐文件加载到内存（仅限小 session，~几千帧）

用法：训练脚本直接用，不需要先 pack（但 pack 后更快）。
"""

import json
import struct
from pathlib import Path
from io import BytesIO

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

PREPROCESSED_ROOT = Path(__file__).resolve().parent.parent / "preprocessed"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SessionBuffer:
    """按需读取 session 帧。优先用 .pak 文件（O(1) seek+read），
    小 session 才全加载到 RAM。"""

    def __init__(self, session_dir: Path):
        self.name = session_dir.name
        self._bytes: dict[int, bytes] = {}        # 小 session 的内存缓存
        self._pak_f = None                         # .pak 文件句柄（按需读）
        self._pak_index: dict[int, tuple] = {}    # fid -> (offset, length)
        self._frame_ids: list[int] = []

        pak_path = session_dir / "frames.pak"
        if pak_path.exists():
            self._init_pak(pak_path)
        else:
            self._init_files(session_dir / "frames")
        self._frame_ids = sorted(self._pak_index.keys()
                                 if self._pak_f
                                 else self._bytes.keys())

    def _init_pak(self, pak_path: Path):
        """解析 .pak 索引，保持文件句柄打开。不加载帧数据！"""
        self._pak_f = open(pak_path, "rb")
        header = self._pak_f.read(12)
        num_frames, index_offset = struct.unpack("<IQ", header)
        self._pak_f.seek(index_offset)
        idx_data = self._pak_f.read(num_frames * 16)
        for i in range(num_frames):
            off = i * 16
            fid, frame_off, frame_len = struct.unpack_from(
                "<IQI", idx_data, off)
            self._pak_index[fid] = (frame_off + 4, frame_len)

    def _init_files(self, frames_dir: Path):
        """无 .pak：逐个文件加载（仅限小 session）。"""
        import warnings
        warnings.warn(f"No .pak found in {frames_dir.parent}, "
                      f"reading individual files (slow). "
                      f"Run: python -m preprocess.pack")
        for p in sorted(frames_dir.glob("*.jpg")):
            fid = int(p.stem)
            self._bytes[fid] = p.read_bytes()

    def __len__(self):
        return len(self._pak_index) if self._pak_f else len(self._bytes)

    def has_frame(self, fid: int) -> bool:
        """检查帧是否存在。"""
        if self._pak_f:
            return fid in self._pak_index
        return fid in self._bytes

    def decode(self, frame_id: int) -> Image.Image:
        """读取并解码一帧。pak 模式按需 seek+read，内存模式读缓存。"""
        if self._pak_f:
            off, lng = self._pak_index[frame_id]
            self._pak_f.seek(off)
            data = self._pak_f.read(lng)
        else:
            data = self._bytes[frame_id]
        return Image.open(BytesIO(data)).convert("RGB")

    @property
    def frame_ids(self):
        return self._frame_ids

    def close(self):
        """关闭文件句柄。"""
        if self._pak_f:
            self._pak_f.close()
            self._pak_f = None


def load_all_sessions(root=PREPROCESSED_ROOT, min_frames=1000,
                      max_total_frames=180000):
    """加载 session 的元数据到内存（不加载帧数据）。

    min_frames: 跳过帧数太少的测试 session
    max_total_frames: 总帧数上限
    """
    candidates = []
    for sess_dir in sorted(root.glob("session_*")):
        has_frames = (sess_dir / "frames").exists()
        has_pak = (sess_dir / "frames.pak").exists()
        if not has_frames and not has_pak:
            continue

        pak_path = sess_dir / "frames.pak"
        if pak_path.exists():
            with open(pak_path, "rb") as f:
                n_frames = struct.unpack_from("<I", f.read(4))[0]
        else:
            labels_path = sess_dir / "labels.json"
            if not labels_path.exists():
                continue
            with open(labels_path, "r") as f:
                import json as _json
                n_frames = len(_json.load(f)["frame_ids"])
        if n_frames >= min_frames:
            candidates.append((n_frames, sess_dir))

    candidates.sort(key=lambda x: x[0], reverse=True)

    buffers = {}
    total_loaded = 0
    import time
    for n_frames, sess_dir in candidates:
        if total_loaded + n_frames > max_total_frames:
            print(f"  (skipping {sess_dir.name} ({n_frames} frames), "
                  f"would exceed {max_total_frames})")
            continue
        print(f"  Loading {sess_dir.name} ({n_frames} frames)...",
              end=" ", flush=True)
        t0 = time.time()
        buf = SessionBuffer(sess_dir)
        t = time.time() - t0
        mem_mb = sum(v[1] for v in buf._pak_index.values()) / 1e6 if buf._pak_f else 0
        print(f"{len(buf)} frames in {t:.1f}s (index only, ~{mem_mb:.0f} MB)",
              flush=True)
        buffers[sess_dir.name] = buf
        total_loaded += len(buf)

    print(f"  Total: {total_loaded} frames in {len(buffers)} sessions",
          flush=True)
    return buffers


def collect_in_memory(buffers: dict, seq_len=4, step=1, max_samples=0,
                      exclude_sessions: set = None, mode: str = None):
    """从 SessionBuffer 字典收集帧堆叠序列。

    mode: None=全部帧, "combat"=只战斗帧, "explore"=只探索帧
          需要先运行 label_combat.py 生成 combat_labels.npy
    """
    # 计算总可用帧数，用于按比例分配 max_samples
    if mode and max_samples:
        total_eligible = 0
        session_frames = {}
        for sess_name in buffers:
            combat_path = PREPROCESSED_ROOT / sess_name / "combat_labels.npy"
            if combat_path.exists():
                cl = np.load(combat_path)
                eligible = int((cl == (1 if mode == "combat" else 0)).sum())
                session_frames[sess_name] = eligible
                total_eligible += eligible
            else:
                session_frames[sess_name] = 0

    sequences = []
    for sess_name, buf in buffers.items():
        if exclude_sessions and sess_name in exclude_sessions:
            continue

        labels_path = PREPROCESSED_ROOT / sess_name / "labels_clean.json"
        if not labels_path.exists():
            labels_path = PREPROCESSED_ROOT / sess_name / "labels.json"
        if not labels_path.exists():
            continue

        # 按模式过滤：加载战斗标签
        if mode:
            combat_path = PREPROCESSED_ROOT / sess_name / "combat_labels.npy"
            if not combat_path.exists():
                continue  # 没有标签的 session 跳过
            combat_labels = np.load(combat_path)
            target_val = 1 if mode == "combat" else 0
            n_target = (combat_labels == target_val).sum()
            print(f"  [{mode}] {sess_name}: {n_target} target frames...", end=" ", flush=True)
            if n_target == 0:
                print("skip", flush=True)
                continue
        else:
            combat_labels = None
            target_val = None

        # 本 session 的 max_samples 配额（按比例）
        if mode and max_samples and total_eligible > 0:
            sess_max = max(1, int(max_samples * session_frames.get(sess_name, 0) / total_eligible))
        else:
            sess_max = max_samples

        with open(labels_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        fids = label_data["frame_ids"]
        btns = label_data["buttons"]
        axes = label_data["axes"]
        n = len(fids)
        if n < seq_len:
            continue

        sess_count = 0
        for i in range(0, n - seq_len + 1, step):
            # 过滤：窗口最后一帧必须匹配目标类型
            if combat_labels is not None:
                last_idx = i + seq_len - 1
                if last_idx < len(combat_labels) and combat_labels[last_idx] != target_val:
                    continue

            seq_fids = [fids[j] for j in range(i, i + seq_len)]
            if all(buf.has_frame(fid) for fid in seq_fids):
                sequences.append((sess_name, seq_fids,
                                  btns[i + seq_len - 1],
                                  axes[i + seq_len - 1]))
                sess_count += 1
                if sess_max and sess_count >= sess_max:
                    break

        if combat_labels is not None:
            print(f"{sess_count} seqs collected", flush=True)

    if mode:
        print(f"  [{mode}] 总计: {len(sequences)} 序列", flush=True)
    return sequences


class InMemoryDataset(Dataset):
    """帧堆叠 Dataset。帧按需从 pak 文件或内存缓存解码。"""

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
