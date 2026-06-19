"""三条血条 + 锁定检测 调试工具 v4（不开AI，纯视觉检测）。

窗口：
  1. 左下角 tkinter 悬浮窗 — 实时数值
  2. "Detection" — 画面 + 检测框
  3. "ROI View" — 每个检测区域的放大截取内容

操作：Q 退出 | S 保存截图到 debug_frame.png + roi_view.png
"""

import sys
import time
import ctypes
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_state import GameState, draw_debug
from controller_overlay import ControllerOverlay

try:
    import mss
except ImportError:
    print("需要 mss: pip install mss")
    sys.exit(1)


def build_roi_view(gs, frame, state):
    """把 6 个检测区域的 ROI 拼接成一张大图，附上原始像素数据。"""
    roi_images = []

    specs = [
        ("fp", "FP蓝", (200, 150, 0)),
        ("hp", "HP红", (0, 200, 0)),
        ("stamina", "体力绿", (0, 200, 200)),
        ("lock_circle", "锁定圈", (255, 255, 255)),
        ("boss_hp", "Boss", (0, 0, 255)),
        ("enemy_hp", "小怪", (0, 165, 255)),
    ]

    for name, label, clr in specs:
        roi, _ = gs._get_roi(frame, name)
        if roi is None or roi.size == 0:
            continue
        rh, rw = roi.shape[:2]

        # 计算这个 ROI 的彩色像素统计
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        stats_str = ""
        if name == "fp":
            blue_mask = cv2.inRange(hsv, (95, 100, 70), (125, 255, 255))
            stats_str = f" blue_px={blue_mask.sum()//255}"
        elif name == "hp":
            red1 = cv2.inRange(hsv, (0, 70, 60), (10, 255, 255))
            red2 = cv2.inRange(hsv, (165, 70, 60), (180, 255, 255))
            red_px = (red1 | red2).sum() // 255
            stats_str = f" red_px={red_px}"
        elif name == "stamina":
            green_mask = cv2.inRange(hsv, (40, 90, 70), (80, 255, 255))
            stats_str = f" green_px={green_mask.sum()//255}"
        elif name == "lock_circle":
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            bright_px = (gray_roi > 210).sum()
            stats_str = f" bright>210={bright_px}"

        # 缩放
        panel_w = 300
        scale = panel_w / rw
        new_h = max(20, int(rh * scale))
        resized = cv2.resize(roi, (panel_w, new_h))

        # 标签
        v = -1
        if name == "fp":
            v = state.get("fp", -1)
        elif name == "hp":
            v = state.get("hp", -1)
        elif name == "stamina":
            v = state.get("stamina", -1)
        elif name == "boss_hp":
            v = state.get("boss_hp", -1)
        elif name == "enemy_hp":
            v = state.get("enemy_hp", -1)
        elif name == "lock_circle":
            v = 1.0 if state.get("locked") else 0.0

        if v >= 0:
            lbl = f"{label} {v:.0%}{stats_str}"
        else:
            lbl = f"{label} --{stats_str}"

        cv2.putText(resized, lbl, (3, new_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, clr, 1)
        roi_images.append(resized)

    if not roi_images:
        return np.zeros((100, 600, 3), dtype=np.uint8)

    # 统一高度
    max_h = max(r.shape[0] for r in roi_images)
    padded = []
    for r in roi_images:
        rh = r.shape[0]
        if rh < max_h:
            pad = np.zeros((max_h - rh, r.shape[1], 3), dtype=np.uint8)
            r = np.vstack([r, pad])
        padded.append(r)

    return np.hstack(padded)


def main():
    sct = mss.MSS()
    monitor = sct.monitors[1]
    w, h = monitor["width"], monitor["height"]

    gs = GameState((w, h))
    overlay = ControllerOverlay()

    print("=" * 55)
    print("  三条血条 + 锁定 检测调试工具 v4")
    print("=" * 55)
    print(f"  屏幕: {w}x{h}")
    print("  [左下角悬浮窗]  实时数值")
    print("  [Detection 窗口] 画面 + 检测框")
    print("  [ROI View 窗口]  放大 ROI + 彩色像素数")
    print("  F12 = 退出 | S = 保存截图")
    print()

    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection", w // 3, h // 3)
    cv2.namedWindow("ROI View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI View", 900, 400)

    running = True
    cnt = 0
    last_print = time.perf_counter()
    fps_t0 = time.perf_counter()
    fps_cnt = 0
    fps = 0

    while running:
        loop_start = time.perf_counter()

        img = np.array(sct.grab(monitor))
        frame = img[:, :, :3].copy()

        try:
            state = gs.detect(frame)
        except Exception as e:
            print(f"\n[错误] 检测失败: {e}")
            state = None

        if state is None:
            state = {"hp": -1, "fp": -1, "stamina": -1, "boss_hp": -1,
                     "enemy_hp": -1, "locked": False, "hp_delta": 0}

        debug_frame = draw_debug(frame, state)
        overlay.update(game_state=state)
        roi_view = build_roi_view(gs, frame, state)

        fps_cnt += 1
        if fps_cnt >= 30:
            now_fps = time.perf_counter()
            fps = 30 / max(0.001, now_fps - fps_t0)
            fps_t0 = now_fps
            fps_cnt = 0

        cv2.putText(debug_frame, f"FPS:{fps:.0f} F12:quit [S]ave",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Detection", debug_frame)
        cv2.imshow("ROI View", roi_view.astype(np.uint8))

        cnt += 1
        now = time.perf_counter()
        if now - last_print >= 0.5:
            last_print = now
            parts = []
            for k, name in [("hp", "HP"), ("fp", "FP"), ("stamina", "体")]:
                v = state.get(k, -1)
                parts.append(f"{name}={v:.0%}" if v >= 0 else f"{name}=--")
            lock_str = "锁:Y" if state.get("locked") else "锁:N"
            boss_str = f"B={state['boss_hp']:.0%}" if state["boss_hp"] >= 0 else "B=无"
            enemy_str = f" E={state['enemy_hp']:.0%}" if state["enemy_hp"] >= 0 else ""
            print(f"\r[{cnt:5d}] {' | '.join(parts)} | {lock_str} | {boss_str}{enemy_str}",
                  end="", flush=True)

        key = cv2.waitKey(1) & 0xFF
        if key == ctypes.windll.user32.GetAsyncKeyState(0x7B) & 0x8000:
            running = False
        if key == ord('s'):
            cv2.imwrite("debug_frame.png", debug_frame)
            cv2.imwrite("roi_view.png", roi_view.astype(np.uint8))
            print(f"\n>>> 截图已保存: debug_frame.png + roi_view.png")

        elapsed = time.perf_counter() - loop_start
        sleep_time = max(0.001, 1/60 - elapsed)
        time.sleep(sleep_time)

    cv2.destroyAllWindows()
    overlay.destroy()
    print("\n退出")


if __name__ == "__main__":
    main()
