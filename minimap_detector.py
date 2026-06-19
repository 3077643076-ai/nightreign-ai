"""小地图敌人检测：找小地图上的红点（敌人标记）。

黑环小地图在屏幕左上角，敌人是红色圆点。
检测到红点 → 说明附近有敌人 → 可以尝试锁定。

用法：
    python minimap_detector.py
    进游戏看终端输出，Q 退出
"""

import sys
import time
import ctypes
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mss


def detect_minimap_enemies(frame):
    """检测小地图上的红点（敌人标记）。

    小地图约在屏幕左上角 1/6 宽 × 1/5 高的区域。
    返回 (是否有敌人, 敌人数量)。
    """
    h, w = frame.shape[:2]
    # 小地图区域：左上角
    mm_w = w // 6
    mm_h = h // 5
    roi = frame[0:mm_h, 0:mm_w]

    # 红色敌人标记：R 高，G/B 低，且比较亮
    b, g, r = (roi[:, :, i].astype(np.float32) for i in range(3))
    # 红色 + 高亮度（红点在暗色地图上很显眼）
    red = (r > 180) & (r > g * 2.0) & (r > b * 2.0)

    # 找红色小圆点
    red_mask = red.astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    enemies = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 红点很小（5-50 像素左右）
        if 5 < area < 200:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # 接近圆形（宽高比接近 1）
            aspect = bw / max(bh, 1)
            if 0.5 < aspect < 2.0:
                enemies.append((x, y, bw, bh))

    return len(enemies) > 0, len(enemies), red_mask


def main():
    sct = mss.MSS()
    monitor = sct.monitors[1]

    print("=" * 50)
    print("  小地图敌人检测")
    print("  进游戏看终端输出")
    print("=" * 50)

    cnt = 0
    while True:
        img = np.array(sct.grab(monitor))
        frame = img[:, :, :3].copy()

        has_enemy, count, _ = detect_minimap_enemies(frame)
        locked = False  # 也可以用之前的 lock circle 检测

        cnt += 1
        if cnt % 15 == 0:
            if has_enemy:
                print(f"  #{cnt} [小地图: {count}个敌人] → 附近有怪，尝试锁定！", flush=True)
            else:
                print(f"  #{cnt} [小地图: 无敌人]", flush=True)

        if ctypes.windll.user32.GetAsyncKeyState(ord('Q')) & 0x8000:
            break


if __name__ == "__main__":
    main()
