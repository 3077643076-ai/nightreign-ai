"""B站视频自动管道：下载 → 抽帧 → 删视频 → 下一个。

用法:
    python -m preprocess.bilibili_pipeline BV1xxx BV1yyy ...
    python -m preprocess.bilibili_pipeline --from-file bv_list.txt

每个 BV:
  1. BBDown 下载视频到 expert_videos/<aid>/
  2. 找到下载的 mp4，抽帧到 preprocessed/expert_<aid>/frames/
  3. 删除原始 mp4 和下载目录
"""

import sys
import subprocess
import shutil
import json
import argparse
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPERT_DIR = Path(__file__).resolve().parent.parent / "expert_videos"
PREPROCESSED_DIR = Path(__file__).resolve().parent.parent / "preprocessed"

FPS = 15
IMG_SIZE = 224
JPEG_QUALITY = 85


def find_mp4(directory: Path) -> list[Path]:
    """递归找所有 mp4 文件。"""
    mp4s = list(directory.glob("**/*.mp4"))
    if not mp4s:
        mp4s = list(directory.glob("**/*.mkv"))
    if not mp4s:
        mp4s = list(directory.glob("**/*.flv"))
    return mp4s


def extract_frames(video_path: Path, out_dir: Path):
    """从视频抽帧到 224x224 JPEG。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    ERROR: Cannot open {video_path.name}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(video_fps / FPS))
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(frames_dir / f"{count:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            count += 1
        frame_idx += 1

    cap.release()
    return count


def process_bv(bv: str):
    """下载 + 处理一个 BV 号。"""
    print(f"\n{'='*50}")
    print(f"[{bv}]")

    # 1. 检查是否已处理过
    existing = list(PREPROCESSED_DIR.glob(f"expert_{bv}*"))
    if existing:
        print(f"  Already processed: {existing[0].name}")
        return

    # 2. 下载
    url = f"https://www.bilibili.com/video/{bv}/"
    print(f"  Downloading...")
    result = subprocess.run(
        ["BBDown", url, "--skip-ai", "--skip-cover", "--skip-subtitle",
         "--dfn-priority", "480P 高清,360P 高清"],
        cwd=str(EXPERT_DIR),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  Download failed: {result.stderr[-200:]}")
        return

    # 3. 找到下载的目录（BBDown 按 aid 建目录）
    download_dirs = sorted(EXPERT_DIR.glob("[0-9]*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not download_dirs:
        print("  No download directory found")
        return
    dl_dir = download_dirs[0]

    mp4s = find_mp4(dl_dir)
    if not mp4s:
        print(f"  No mp4 found in {dl_dir}")
        return

    # 4. 抽帧
    out_name = f"expert_{bv}"
    out_dir = PREPROCESSED_DIR / out_name

    total_frames = 0
    for mp4 in mp4s:
        print(f"  Extracting frames from {mp4.name}...")
        n = extract_frames(mp4, out_dir)
        total_frames += n
        print(f"    {n} frames")

    # 5. 写元信息
    with open(out_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump({
            "bv": bv,
            "source": url,
            "frames": total_frames,
            "fps": FPS,
            "img_size": IMG_SIZE,
        }, f, ensure_ascii=False)

    # 6. 删除原始下载
    shutil.rmtree(dl_dir, ignore_errors=True)
    # 也清理可能的 m4a 等单独文件
    for f in EXPERT_DIR.glob("*.m4a"):
        f.unlink(missing_ok=True)

    print(f"  Done: {total_frames} frames, video deleted, saved to {out_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bvs", nargs="*", help="BV numbers to process")
    parser.add_argument("--from-file", type=str, help="Read BV list from file")
    args = parser.parse_args()

    bvs = args.bvs
    if args.from_file:
        with open(args.from_file, "r") as f:
            for line in f:
                bv = line.strip().split("#")[0].strip()  # 支持 # 注释
                if bv:
                    bvs.append(bv)

    if not bvs:
        print("Usage: python -m preprocess.bilibili_pipeline BV1xxx BV2xxx ...")
        return

    print(f"Processing {len(bvs)} videos")
    for bv in bvs:
        try:
            process_bv(bv)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # 汇总
    print(f"\n{'='*50}")
    total_frames = 0
    for d in sorted(PREPROCESSED_DIR.glob("expert_*")):
        info_path = d / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            total_frames += info.get("frames", 0)
            print(f"  {d.name}: {info.get('frames', 0)} frames")

    print(f"\nTotal expert frames: {total_frames}")


if __name__ == "__main__":
    main()
