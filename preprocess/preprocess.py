"""一次性预处理：把所有 session 的帧缩放到 224x224，存为小 JPEG。

用法：
    python -c "import sys; sys.path.insert(0,'X:/dev/game-ai-agent'); from preprocess.preprocess import preprocess; preprocess()"

逐 session 处理，防止 Python 3.13 大内存崩溃。
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
CHUNK_SIZE = 5000  # 每处理 N 帧释放一次内存


def process_session(sess: Path):
    inputs_path = sess / "inputs.jsonl"
    frames_dir = sess / "frames"
    if not inputs_path.exists() or not frames_dir.exists():
        return 0

    out_dir = PREPROCESSED_DIR / sess.name
    out_frames = out_dir / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)

    lines = []
    with open(inputs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line)

    frame_ids = []
    btn_list = []
    axis_list = []
    count = 0

    for i, line in enumerate(lines):
        d = json.loads(line)
        fid = d["frame"]
        dst = out_frames / f"{fid:06d}.jpg"

        if not dst.exists():
            src = str(frames_dir / f"{fid:06d}.jpg")
            img = cv2.imread(src)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        frame_ids.append(fid)
        btns, axes = _normalize(d["buttons"], d["axes"])
        btn_list.append(btns)
        axis_list.append(axes)
        count += 1

        if count % CHUNK_SIZE == 0:
            print(f"  {sess.name}: {count}/{len(lines)}")

    # 存标签
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({
            "frame_ids": frame_ids,
            "buttons": btn_list,
            "axes": axis_list,
        }, f, ensure_ascii=False)

    return count


def preprocess():
    sessions = sorted(DATA_ROOT.glob("session_*"))
    if not sessions:
        print("No sessions found.")
        return

    total = 0
    for sess in sessions:
        n = process_session(sess)
        print(f"  {sess.name}: {n} frames saved")
        total += n

    print(f"\nDone. Total: {total} frames in {PREPROCESSED_DIR}")


if __name__ == "__main__":
    preprocess()
