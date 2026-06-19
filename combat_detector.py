"""战斗目标检测器：找画面中的白色锁定标记，判断是否可以锁定敌人。

用法：
    python combat_detector.py  # 实时显示检测结果
    按 Q 退出

先在训练场开游戏，然后运行这个脚本，看看能不能检测到怪身上的白点。
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
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    print("需要 mss: pip install mss")
    sys.exit(1)


def main():
    sct = mss.MSS()
    monitor = sct.monitors[1]
    print(f"屏幕: {monitor['width']}x{monitor['height']}")
    print("Q = 退出 | 白色区域会被标记出来")
    print("靠近敌人，看中间区域能不能检测到白点\n")

    while True:
        # 截图
        img = np.array(sct.grab(monitor))
        frame = img[:, :, :3].copy()  # BGRA → BGR, 必须 copy 才能画框

        h, w = frame.shape[:2]

        # 只分析屏幕中间区域（敌人通常在中间）
        cx, cy = w // 2, h // 2
        roi_w, roi_h = w // 3, h // 3
        x1 = cx - roi_w // 2
        y1 = cy - roi_h // 2
        x2 = cx + roi_w // 2
        y2 = cy + roi_h // 2
        roi = frame[y1:y2, x1:x2]

        # 找白色区域（锁定标记是白色的）
        # 白色 = R>200, G>200, B>200，且比较亮的像素
        white_mask = cv2.inRange(roi, np.array([200, 200, 200]), np.array([255, 255, 255]))

        # 形态学处理：去掉噪点，保留团块
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 筛选足够大的白块（排除小噪点）
        found = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 30:  # 至少 30 像素
                # 在原图上画框
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x1 + rx, y1 + ry),
                              (x1 + rx + rw, y1 + ry + rh), (0, 255, 0), 2)
                if area > 60:  # 大块 = 很可能可以锁定
                    found = True

        # 画 ROI 框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        # 状态文字
        status = "LOCK-ON AVAILABLE!" if found else "no target"
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(frame, status, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 每 30 帧打印状态
        if not hasattr(main, "_cnt"):
            main._cnt = 0
            main._last_print = time.perf_counter()
        main._cnt += 1
        if main._cnt % 30 == 0:
            now = time.perf_counter()
            dt = now - main._last_print
            main._last_print = now
            status_str = "TARGET FOUND!" if found else "no target"
            print(f"\r[{main._cnt}] {status_str}  "
                  f"({len([c for c in contours if cv2.contourArea(c)>30])} white blobs)  "
                  f"fps={30/dt:.0f}", end="", flush=True)

        # 每 5 秒保存一张截图方便看
        if main._cnt % 75 == 0:
            cv2.imwrite("detector_debug.png", frame)
            if found:
                print("\n  -> 截图已保存 detector_debug.png", end="", flush=True)

        # Q 键退出
        if ctypes.windll.user32.GetAsyncKeyState(ord('Q')) & 0x8000:
            break

    print("\n退出")


if __name__ == "__main__":
    main()
