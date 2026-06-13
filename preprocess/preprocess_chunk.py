"""分块预处理：每次只处理 CHUNK_SIZE 帧，内存友好。

用法（bash 循环）:
    for start in $(seq 0 5000 174000); do
        python preprocess_chunk.py session_20260613_122723 $start
    done
"""

import sys
import json
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recorder.config import DATA_ROOT
from preprocess.dataset import _normalize

PREPROCESSED_DIR = Path(__file__).resolve().parent.parent / "preprocessed"
IMG_SIZE = 224
CHUNK_SIZE = 5000


def process_chunk(session_name: str, start: int = 0):
    sess = DATA_ROOT / session_name
    inputs_path = sess / "inputs.jsonl"
    frames_dir = sess / "frames"
    if not inputs_path.exists():
        print(f"ERROR: {inputs_path} not found")
        sys.exit(1)

    out_dir = PREPROCESSED_DIR / session_name
    out_frames = out_dir / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)

    # 读所有行
    all_lines = []
    with open(inputs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_lines.append(line)

    end = min(start + CHUNK_SIZE, len(all_lines))
    chunk_lines = all_lines[start:end]

    frame_ids = []
    btn_list = []
    axis_list = []
    count = 0

    for line in chunk_lines:
        d = json.loads(line)
        fid = d["frame"]
        dst = out_frames / f"{fid:06d}.jpg"

        if not dst.exists():
            src = str(frames_dir / f"{fid:06d}.jpg")
            img = cv2.imread(src)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        frame_ids.append(fid)
        btns, axes = _normalize(d["buttons"], d["axes"])
        btn_list.append(btns)
        axis_list.append(axes)
        count += 1

    # 存当前块的标签
    chunk_dir = out_dir / "labels_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    with open(chunk_dir / f"chunk_{start:06d}.json", "w", encoding="utf-8") as f:
        json.dump({
            "start": start, "end": end, "total": len(all_lines),
            "frame_ids": frame_ids, "buttons": btn_list, "axes": axis_list,
        }, f, ensure_ascii=False)

    print(f"  {session_name}: {start}-{end} / {len(all_lines)} ({count} frames)")

    # 如果是最后一块，合并所有块
    if end >= len(all_lines):
        merge_chunks(out_dir, len(all_lines))
        print(f"  {session_name}: ALL DONE, labels.json merged")


def merge_chunks(out_dir: Path, total: int):
    all_fids = []
    all_btns = []
    all_axes = []
    chunk_dir = out_dir / "labels_chunks"
    for cf in sorted(chunk_dir.glob("chunk_*.json")):
        with open(cf) as f:
            d = json.load(f)
        all_fids.extend(d["frame_ids"])
        all_btns.extend(d["buttons"])
        all_axes.extend(d["axes"])

    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({
            "frame_ids": all_fids, "buttons": all_btns, "axes": all_axes,
        }, f, ensure_ascii=False)

    # 清理分块文件
    import shutil
    shutil.rmtree(chunk_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_chunk.py <session_name> [start]")
        sys.exit(1)
    session_name = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    process_chunk(session_name, start)
