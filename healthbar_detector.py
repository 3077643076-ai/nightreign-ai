"""敌人血条检测 + 白点锁定检测 = 判断是否应该战斗。

检测逻辑：
  1. 屏幕上方区域找红色血条 → 有敌人在附近
  2. 屏幕中间区域找白色锁定圈 → 已锁定敌人
  3. 有血条但没锁定 → 应该按 RS 锁定
  4. 已锁定 → 让模型战斗

用法：
    python healthbar_detector.py
    进训练场靠近敌人，看终端输出
"""

import sys
import time
import ctypes
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import mss
except ImportError:
    print("pip install mss")
    sys.exit(1)


def detect_healthbar(frame):
    """检测屏幕上方是否有敌人血条（红色）。"""
    h, w = frame.shape[:2]
    # 血条通常在屏幕上方 1/4 区域
    roi = frame[h // 8 : h // 3, w // 4 : 3 * w // 4]

    # 红色血条：R 通道高，G/B 通道低
    # 在 BGR 格式中：BGR → 红色是 B 低 G 低 R 高
    b, g, r = roi[:, :, 0].astype(np.float32), roi[:, :, 1].astype(np.float32), roi[:, :, 2].astype(np.float32)

    # 红色像素：R > 150 且 R 比 G 和 B 都高很多
    red = (r > 150) & (r > g * 1.5) & (r > b * 1.5)
    red_count = red.sum()

    # 找红色区域的轮廓
    red_mask = red.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找足够大的红色区域（血条）
    bars = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        # 血条特征：宽度远大于高度（长方形）
        aspect = bw / max(bh, 1)
        if area > 200 and aspect > 3:
            bars.append((x, y, bw, bh))

    return len(bars) > 0, bars


def detect_lock_circle(frame):
    """检测画面中间是否有白色锁定圈（已经锁定敌人）。"""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    rw, rh = w // 3, h // 3
    roi = frame[cy - rh // 2 : cy + rh // 2, cx - rw // 2 : cx + rw // 2]

    white = cv2.inRange(roi, np.array([180, 180, 180]), np.array([255, 255, 255]))
    kernel = np.ones((2, 2), np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 40:
            return True
    return False


def main():
    sct = mss.MSS()
    monitor = sct.monitors[1]

    print("=" * 50)
    print("  血条 + 锁定圈 检测器")
    print("  进训练场靠近敌人")
    print("=" * 50)

    cnt = 0
    while True:
        img = np.array(sct.grab(monitor))
        frame = img[:, :, :3].copy()

        hp, bars = detect_healthbar(frame)
        locked = detect_lock_circle(frame)

        cnt += 1
        if cnt % 15 == 0:
            status = ""
            if hp:
                status += f"[血条: {len(bars)}个] "
            else:
                status += "[无血条] "
            if locked:
                status += "[已锁定] → 战斗模型"
            elif hp:
                status += "[未锁定] → 按RS！"
            else:
                status += "[无目标] → 跑图"
            print(f"\r#{cnt} {status}  ", end="", flush=True)

        # Q 退出
        if ctypes.windll.user32.GetAsyncKeyState(ord('Q')) & 0x8000:
            break

    print("\n退出")


if __name__ == "__main__":
    main()
