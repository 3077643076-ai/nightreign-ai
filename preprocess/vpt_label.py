"""VPT 视频消化：从大佬 mp4 视频提取帧 + 模型伪标注。

用法:
    python -m preprocess.vpt_label          # 处理所有视频
    python -m preprocess.vpt_label --dry-run  # 只检查视频信息，不推理

每段视频按 15fps 抽帧 → 224x224 → 模型推理 → 伪标注存储。
"""

import sys
import json
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.bc_model import BCModel
from preprocess.preprocessed_dataset import IMAGENET_MEAN, IMAGENET_STD
from preprocess.dataset import BUTTON_NAMES, AXIS_NAMES

VIDEO_DIR = Path(__file__).resolve().parent.parent / "expert_videos"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "preprocessed" / "expert_pseudo"
CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "best.pt"

FPS = 15
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.8  # 按钮概率 >0.8 或 <0.2 才算"置信"


def extract_frames(video_path: Path, out_dir: Path, dry_run=False):
    """抽帧到 out_dir/frames/，返回帧数。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / max(video_fps, 1)
    step = max(1, int(video_fps / FPS))

    frames_dir = out_dir / "frames"
    if not dry_run:
        frames_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            if not dry_run:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(str(frames_dir / f"{count:06d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            count += 1

        frame_idx += 1
        if count % 5000 == 0 and count > 0:
            print(f"    {count} frames...")

    cap.release()

    info = {
        "video": video_path.name,
        "video_fps": video_fps,
        "total_frames": total_frames,
        "duration_sec": round(duration, 1),
        "extracted_frames": count,
        "target_fps": FPS,
    }
    return count, info


def load_model(device):
    model = BCModel().to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def pseudo_label(frames_dir: Path, model, device):
    """对已抽取的帧做伪标注。返回 (buttons, axes, confidences) 列表。"""
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        return [], [], []

    btns_all = []
    axes_all = []
    confs_all = []

    batch_size = 64
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is None:
                tensors.append(None)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            tensors.append(torch.from_numpy(img).permute(2, 0, 1))

        valid = [(j, t) for j, t in enumerate(tensors) if t is not None]
        if not valid:
            continue

        indices, batch_tensors = zip(*valid)
        batch = torch.stack(batch_tensors).to(device)

        btn_logits, axes_pred = model(batch)
        btn_prob = torch.sigmoid(btn_logits).cpu().numpy()
        axes_val = axes_pred.cpu().numpy()

        for k in range(len(batch_paths)):
            btns_all.append(np.zeros(17, dtype=np.float32))
            axes_all.append(np.zeros(6, dtype=np.float32))
            confs_all.append(np.zeros(17, dtype=np.float32))

        for k, j in enumerate(indices):
            prob = btn_prob[k]
            btns_all[j] = (prob >= 0.5).astype(np.float32)
            axes_all[j] = axes_val[k]
            # 置信度：远离 0.5 的程度
            confs_all[j] = np.abs(prob - 0.5) * 2

        if (i // batch_size) % 50 == 0 and i > 0:
            print(f"    pseudo-label: {i}/{len(frame_paths)}")

    return btns_all, axes_all, confs_all


def process_video(video_path: Path, model, device, dry_run=False, idx=0):
    """处理单个视频：抽帧 + 伪标注。"""
    # 用数字编号避免控制台编码问题
    out_dir = OUTPUT_DIR / video_path.stem
    print(f"\n[Video #{idx}] {video_path.suffix}")

    # 抽帧
    count, info = extract_frames(video_path, out_dir, dry_run)
    print(f"  Frames: {count} | Duration: {info['duration_sec']}s")
    if count == 0:
        return

    if dry_run:
        return

    # 伪标注
    print("  Pseudo-labeling...")
    btns, axes, confs = pseudo_label(out_dir / "frames", model, device)

    # 统计高置信帧
    high_conf_frames = 0
    high_conf_actions = 0
    for c in confs:
        high = c > CONFIDENCE_THRESHOLD
        if high.any():
            high_conf_frames += 1
            high_conf_actions += high.sum()

    info["pseudo_labeled_frames"] = len(btns)
    info["high_confidence_frames"] = high_conf_frames
    info["high_confidence_actions"] = int(high_conf_actions)

    # 存伪标注
    with open(out_dir / "pseudo_labels.json", "w", encoding="utf-8") as f:
        json.dump({
            "info": info,
            "button_names": list(BUTTON_NAMES),
            "axis_names": list(AXIS_NAMES),
            "buttons": [b.tolist() for b in btns],
            "axes": [a.tolist() for a in axes],
            "confidences": [c.tolist() for c in confs],
        }, f, ensure_ascii=False)

    print(f"  High-confidence frames: {high_conf_frames}/{len(btns)} "
          f"({info.get('high_confidence_actions', 0)} actions)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="only check videos, no processing")
    parser.add_argument("--video", type=str, help="filter videos by name")
    args = parser.parse_args()

    videos = list(VIDEO_DIR.glob("*.mp4"))
    for d in VIDEO_DIR.iterdir():
        if d.is_dir():
            videos.extend(d.glob("*.mp4"))

    if not videos:
        print("No videos found in", VIDEO_DIR)
        return

    print(f"Found {len(videos)} videos")
    for i, v in enumerate(videos):
        size_mb = v.stat().st_size / 1e6
        print(f"  [{i}] {size_mb:.0f}MB {v.suffix}")

    if args.video:
        videos = [v for v in videos if args.video in v.name]
        if not videos:
            print(f"No video matching '{args.video}'")
            return

    if args.dry_run:
        for i, v in enumerate(videos):
            _, info = extract_frames(v, OUTPUT_DIR / v.stem, dry_run=True)
            print(f"  [{i}] fps={info['video_fps']:.1f} frames={info['total_frames']} dur={info['duration_sec']}s → {info['extracted_frames']} @{FPS}fps")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = None
    if not args.dry_run:
        if CHECKPOINT.exists():
            model = load_model(device)
            print(f"Model loaded: {CHECKPOINT}")
        else:
            print(f"WARNING: No checkpoint at {CHECKPOINT}, will only extract frames")

    for i, v in enumerate(videos):
        process_video(v, model, device, dry_run=args.dry_run, idx=i)

    print("\nDone.")


if __name__ == "__main__":
    main()
