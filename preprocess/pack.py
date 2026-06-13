"""打包工具：把一个 session 的 JPEG 帧打包成单个 .pak 文件。

格式:
    [header]  num_frames: u32, index_offset: u64
    [frames]  repeated: jpg_len: u32, jpg_data: bytes
    [index]   repeated: frame_id: u32, offset: u64, length: u32

打包后训练时零随机 I/O — 只需打开一个文件，seek 到偏移量读取。

用法:
    python -m preprocess.pack              # 打包所有 session
"""

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PREPROCESSED_ROOT = Path(__file__).resolve().parent.parent / "preprocessed"


def pack_session(session_dir: Path):
    """将 session_dir/frames/*.jpg 打包为 session_dir/frames.pak。"""
    frames_dir = session_dir / "frames"
    if not frames_dir.exists():
        return 0

    pak_path = session_dir / "frames.pak"
    jpgs = sorted(frames_dir.glob("*.jpg"))
    if not jpgs:
        return 0

    # 预计算每个 frame 的 id
    entries = [(int(p.stem), p) for p in jpgs]

    with open(pak_path, "wb") as f:
        # 头部占位（写完再回填）
        header_pos = f.tell()
        f.write(struct.pack("<IQ", 0, 0))  # placeholder

        # 写帧数据
        offsets = []
        lengths = []
        frame_ids = []
        for fid, p in entries:
            data = p.read_bytes()
            offsets.append(f.tell())
            lengths.append(len(data))
            frame_ids.append(fid)
            f.write(struct.pack("<I", len(data)))
            f.write(data)

        # 写索引
        index_offset = f.tell()
        for fid, off, lng in zip(frame_ids, offsets, lengths):
            f.write(struct.pack("<IQI", fid, off, lng))

        # 回填头部
        f.seek(header_pos)
        f.write(struct.pack("<IQ", len(frame_ids), index_offset))

    # 计算压缩比
    total_raw = len(frame_ids) * 224 * 224 * 3
    total_packed = pak_path.stat().st_size
    print(f"  {session_dir.name}: {len(frame_ids)} frames, "
          f"{total_packed / 1e6:.1f} MB (vs {total_raw / 1e6:.0f} MB raw, "
          f"{total_raw / max(total_packed, 1):.1f}x)")

    return len(frame_ids)


def pack_all():
    sessions = sorted(PREPROCESSED_ROOT.glob("session_*"))
    total = 0
    for sess in sessions:
        n = pack_session(sess)
        total += n
    print(f"\nTotal: {total} frames packed")


class PakReader:
    """读取 .pak 文件，O(1) 随机访问任意帧。"""

    def __init__(self, pak_path: Path):
        with open(pak_path, "rb") as f:
            header = f.read(12)
            num_frames, index_offset = struct.unpack("<IQ", header)

            f.seek(index_offset)
            index_data = f.read(num_frames * 16)  # 4+8+4 = 16 bytes per entry
            self._index = {}
            for i in range(num_frames):
                off = i * 16
                fid, frame_off, frame_len = struct.unpack(
                    "<IQI", index_data[off:off + 16])
                self._index[fid] = (frame_off, frame_len)

            self._f = open(pak_path, "rb")  # 保持文件打开
            self.frame_ids = sorted(self._index.keys())

    def read(self, frame_id: int) -> bytes:
        """读取一帧的 JPEG 字节。"""
        off, lng = self._index[frame_id]
        self._f.seek(off + 4)  # 跳过 length prefix
        return self._f.read(lng)

    def close(self):
        self._f.close()

    def __len__(self):
        return len(self._index)


if __name__ == "__main__":
    pack_all()
