"""HUD 分析工具：截一张图，分析所有检测区域。

用法：打开游戏，让 HUD（血条/蓝条/体力条）在屏幕上可见，然后运行：
    python analyze_hud.py

会生成：
  - capture.png          完整截图
  - roi_<区域名>.png     每个检测区域的放大图
  - roi_<区域名>_mask.png 颜色掩码（红色/蓝色/绿色像素用白色标记）
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_state import GameState
from capture import capture_game


def analyze_roi(frame, gs, name, color_name, hsv_ranges):
    """详细分析一个 ROI 区域。"""
    roi, (y1, y2, x1, x2) = gs._get_roi(frame, name)
    rh, rw = roi.shape[:2]
    if roi.size == 0:
        print(f"\n{'='*60}")
        print(f"  {name} ({color_name}): ROI 为空！")
        return

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 颜色掩码
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(low), np.array(high))

    # 形态学去噪
    kernel = np.ones((2, 2), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 每列统计
    col_color = mask_clean.sum(axis=0)  # 每列彩色像素数
    col_has = (col_color > 0).sum()  # 有彩色像素的列数

    print(f"\n{'='*60}")
    print(f"  {name} — {color_name}")
    print(f"  位置: ({x1},{y1})-({x2},{y2})  尺寸: {rw}x{rh}")
    print(f"  灰度: 均值={gray.mean():.0f}  中位={np.median(gray):.0f}  范围=[{gray.min()},{gray.max()}]")
    print(f"  彩色像素: {mask.sum()//255} 个 (去噪后 {mask_clean.sum()//255} 个)")
    print(f"  有彩色像素的列: {col_has}/{rw}")
    if col_has > 0:
        color_cols = np.where(col_color > 0)[0]
        print(f"  彩色列范围: [{color_cols.min()}, {color_cols.max()}]")
        print(f"  每列最多彩色像素: {col_color.max()} (行高 {rh})")

    # 找暗像素列（容器边界用）
    col_dark_ratio = (gray < 50).sum(axis=0) / max(1, gray.shape[0])
    dark_cols = np.where(col_dark_ratio > 0.25)[0]
    if len(dark_cols) > 0:
        print(f"  暗列(>25%深度)范围: [{dark_cols.min()}, {dark_cols.max()}]")

    # 测填充率
    calibrated = gs._calibrate_container(roi, color_name)
    if calibrated is not None:
        bar_l, bar_r = calibrated
        fill = gs._measure_fill_in_container(roi, color_name, bar_l, bar_r)
        print(f"  容器: [{bar_l}, {bar_r}] 宽={bar_r-bar_l}  填充率: {fill:.1%}")
    else:
        print(f"  容器校准失败")

    # 保存图片
    cv2.imwrite(f"roi_{name}.png", roi)
    cv2.imwrite(f"roi_{name}_mask.png", mask_clean)
    # 保存掩码叠加图
    overlay = roi.copy()
    overlay[mask_clean > 0] = (0, 255, 255)
    cv2.imwrite(f"roi_{name}_overlay.png", overlay)
    print(f"  已保存: roi_{name}.png / roi_{name}_mask.png / roi_{name}_overlay.png")


def main():
    print("正在截取游戏窗口...")
    frame, (w, h) = capture_game()
    cv2.imwrite("capture.png", frame)
    print(f"屏幕: {w}x{h}")

    gs = GameState((w, h))
    state = gs.detect(frame)

    print(f"\n检测结果:")
    print(f"  HP:      {state['hp']:.1%}" if state['hp'] >= 0 else "  HP:      --")
    print(f"  FP:      {state['fp']:.1%}" if state['fp'] >= 0 else "  FP:      --")
    print(f"  体力:    {state['stamina']:.1%}" if state['stamina'] >= 0 else "  体力:    --")
    print(f"  锁定:    {state['locked']}")
    print(f"  Boss HP: {state['boss_hp']:.1%}" if state['boss_hp'] >= 0 else "  Boss HP: 无")
    print(f"  小怪 HP: {state['enemy_hp']:.1%}" if state['enemy_hp'] >= 0 else "  小怪 HP: 无")

    # 详细分析每个 ROI
    analyze_roi(frame, gs, "hp", "red", gs.COLORS["red"])
    analyze_roi(frame, gs, "fp", "blue", gs.COLORS["blue"])
    analyze_roi(frame, gs, "stamina", "green", gs.COLORS["green"])

    # 锁定圈分析
    print(f"\n{'='*60}")
    print(f"  锁定圈 (lock_circle)")
    roi, (y1, y2, x1, x2) = gs._get_roi(frame, "lock_circle")
    rh, rw = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 3 <= area <= 100:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(1, bh)
            if 0.6 < aspect < 1.6:
                cx, cy = x + bw // 2, y + bh // 2
                r = max(bw, bh) // 2 + 3
                ring_y1 = max(0, cy - r)
                ring_y2 = min(gray.shape[0], cy + r)
                ring_x1 = max(0, cx - r)
                ring_x2 = min(gray.shape[1], cx + r)
                ring = gray[ring_y1:ring_y2, ring_x1:ring_x2]
                ring_mean = ring.mean()
                if ring_mean < 140:
                    valid.append((area, cx, cy))
    print(f"  位置: ({x1},{y1})-({x2},{y2})  尺寸: {rw}x{rh}")
    print(f"  高亮(>210)像素: {(gray > 210).sum()} 个")
    print(f"  有效光点候选: {len(valid)} 个")
    for i, (area, cx, cy) in enumerate(valid):
        print(f"    #{i}: area={area:.0f} 中心=({cx},{cy})")

    cv2.imwrite("roi_lock_circle.png", roi)
    cv2.imwrite("roi_lock_bright.png", bright)
    print(f"  已保存: roi_lock_circle.png / roi_lock_bright.png")

    print(f"\n{'='*60}")
    print("分析完成！请把生成的 PNG 文件发给 Claude 查看。")
    print("主要看 roi_*_overlay.png — 黄色部分就是被检测到的彩色像素。")


if __name__ == "__main__":
    main()
