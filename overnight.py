"""过夜自动脚本：B站视频 → 打包 → 训练。

步骤:
  1. 逐条处理 bv_list.txt 中的视频（下载 → 抽帧 → 删视频）
  2. 打包所有 session（包括专家视频）为 .pak 文件
  3. 训练帧堆叠模型

用法:
    python overnight.py
"""

import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERT_DIR = ROOT / "expert_videos"


def step(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def run_bv_pipeline():
    """逐条处理 BV 列表。"""
    bv_file = ROOT / "bv_list.txt"
    if not bv_file.exists():
        print("No bv_list.txt, skipping video pipeline")
        return

    bvs = []
    with open(bv_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                bv = line.split()[0]  # handle "BVxxx # comment"
                bvs.append(bv)

    print(f"{len(bvs)} BVs to process")

    for i, bv in enumerate(bvs):
        print(f"\n--- [{i+1}/{len(bvs)}] {bv} ---")
        url = f"https://www.bilibili.com/video/{bv}/"

        # 检查是否已处理
        existing = list((ROOT / "preprocessed").glob(f"expert_{bv}*"))
        if existing:
            print(f"  Already done: {existing[0].name}")
            continue

        # 记录下载前的 mp4 列表
        before = set(EXPERT_DIR.glob("*.mp4"))

        # 下载
        print(f"  Downloading...")
        result = subprocess.run(
            ["BBDown", url, "--skip-ai", "--skip-cover", "--skip-subtitle"],
            cwd=str(EXPERT_DIR), timeout=600,
        )
        if result.returncode != 0:
            print(f"  Download FAILED (exit {result.returncode})")
            continue

        # 找到新 mp4
        after = set(EXPERT_DIR.glob("*.mp4"))
        new_mp4s = after - before
        if not new_mp4s:
            print("  No new mp4 found — might have failed silently")
            continue

        # 抽帧
        import cv2
        for mp4 in new_mp4s:
            print(f"  Extracting frames from {mp4.name[:50]}...")
            out_dir = ROOT / "preprocessed" / f"expert_{bv}"
            frames_dir = out_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(mp4))
            if not cap.isOpened():
                print(f"    Cannot open, skipping")
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            step = max(1, int(video_fps / 15))
            count = 0
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    frame = cv2.resize(frame, (224, 224))
                    cv2.imwrite(str(frames_dir / f"{count:06d}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                    count += 1
                frame_idx += 1
            cap.release()

            # 删视频
            mp4.unlink()
            print(f"    {count} frames extracted, video deleted")

        # 清理残留的 m4a 等
        for f in EXPERT_DIR.glob("*.m4a"):
            f.unlink(missing_ok=True)

    print("\nVideo pipeline done.")


def run_pack():
    """打包所有帧为 .pak。"""
    print("Packing frames into .pak files...")
    subprocess.run([sys.executable, "-m", "preprocess.pack"], cwd=str(ROOT),
                   timeout=600)


def run_train():
    """训练帧堆叠模型。"""
    print("Starting training...")
    subprocess.run([sys.executable, "-m", "train.train_mem"], cwd=str(ROOT),
                   timeout=86400)  # 24 hour timeout


def main():
    t0 = time.time()

    step("Step 1/3: Bilibili Video Pipeline")
    try:
        run_bv_pipeline()
    except Exception as e:
        print(f"Video pipeline error: {e}")

    step("Step 2/3: Pack frames")
    try:
        run_pack()
    except Exception as e:
        print(f"Pack error: {e}")

    step("Step 3/3: Train stacked model")
    try:
        run_train()
    except Exception as e:
        print(f"Train error: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Overnight pipeline finished in {elapsed/60:.0f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
