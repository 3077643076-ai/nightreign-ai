"""敌人血条检测器 — 锁定后小血条。

小怪血条出现在屏幕上方中间区域，只有锁定时才显示。
大小和形状与玩家 HP 条类似，但位置不同。
用红色比值法检测。
"""

import cv2
import numpy as np
from perception.base import BarDetector


class EnemyHPDetector(BarDetector):
    """敌人血条检测器。

    ROI: 屏幕中上方 y=0.075~0.20, x=0.10~0.90（宽范围覆盖）
    只在锁定时才触发检测。
    """

    RED_RATIO_THRESHOLD = 0.40
    MIN_RED_COL_FRAC = 0.25
    MIN_BAR_WIDTH = 15

    def __init__(self):
        super().__init__(
            name="enemy_hp",
            roi_fractions=(0.075, 0.20, 0.10, 0.90),
            ema_alpha=0.82,
        )

    def detect(self, frame):
        """检测敌人血条百分比。

        在 ROI 内扫描找红色水平条，返回红色占比。
        """
        roi = self.extract_roi(frame)
        if roi.size == 0:
            return -1.0, {}

        h, w = roi.shape[:2]

        # ── 红色比值法 ──
        b = roi[:, :, 0].astype(np.float32)
        g = roi[:, :, 1].astype(np.float32)
        r = roi[:, :, 2].astype(np.float32)
        total = r + g + b + 1e-6
        red_ratio = r / total
        red_mask = red_ratio > self.RED_RATIO_THRESHOLD

        # ── 找红色最密集的水平行（血条所在行） ──
        row_red_count = red_mask.sum(axis=1)  # 每行红色像素数
        if row_red_count.max() < 5:
            return -1.0, {
                "roi": roi,
                "red_mask": red_mask,
            }

        # 找红色最密集的连续行段 = 血条的垂直位置
        best_row_start, best_row_end = 0, 0
        cur_start = -1
        for i in range(h):
            if row_red_count[i] > 0:
                if cur_start < 0:
                    cur_start = i
            else:
                if cur_start >= 0:
                    if i - cur_start > best_row_end - best_row_start:
                        best_row_start, best_row_end = cur_start, i
                    cur_start = -1
        if cur_start >= 0:
            if h - cur_start > best_row_end - best_row_start:
                best_row_start, best_row_end = cur_start, h

        bar_h = best_row_end - best_row_start
        if bar_h < 3 or bar_h > 40:
            return -1.0, {
                "roi": roi,
                "red_mask": red_mask,
                "bar_row_start": best_row_start,
                "bar_row_end": best_row_end,
            }

        # ── 在找到的条内分析红色填充 ──
        bar_region = red_mask[best_row_start:best_row_end, :]
        col_red_count = bar_region.sum(axis=0)
        min_red_per_col = max(1, int(bar_h * self.MIN_RED_COL_FRAC))
        col_filled = col_red_count >= min_red_per_col

        # 找最长连续红段
        best_start, best_end = 0, 0
        cur_start = -1
        for i in range(w):
            if col_filled[i]:
                if cur_start < 0:
                    cur_start = i
            else:
                if cur_start >= 0:
                    if i - cur_start > best_end - best_start:
                        best_start, best_end = cur_start, i
                    cur_start = -1
        if cur_start >= 0:
            if w - cur_start > best_end - best_start:
                best_start, best_end = cur_start, w

        bar_width = best_end - best_start
        if bar_width < self.MIN_BAR_WIDTH:
            return -1.0, {
                "roi": roi,
                "red_mask": red_mask,
                "bar_width": bar_width,
            }

        # 自动校准
        self.update_max_fill(bar_width)
        max_fill = self.get_max_fill()
        if max_fill < self.MIN_BAR_WIDTH:
            return -1.0, {
                "roi": roi,
                "red_mask": red_mask,
                "bar_width": bar_width,
                "max_fill": max_fill,
            }

        # 敌人血量 = 当前红段宽度 / 最大观察宽度
        raw_value = bar_width / max_fill
        raw_value = max(0.0, min(1.0, raw_value))

        smooth_value = self.smooth(raw_value)

        debug = {
            "roi": roi,
            "red_mask": red_mask,
            "col_red_count": col_red_count,
            "bar_start": best_start,
            "bar_end": best_end,
            "bar_width": bar_width,
            "max_fill": max_fill,
            "bar_row_start": best_row_start,
            "bar_row_end": best_row_end,
            "raw_value": raw_value,
        }

        return smooth_value, debug

    def get_debug_image(self, roi, debug_info):
        """生成调试图像：灰度 ROI + 红色掩码叠加。"""
        if roi is None or roi.size == 0:
            return np.zeros((60, 300), dtype=np.uint8)

        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 画血条行范围
        bar_rs = debug_info.get("bar_row_start")
        bar_re = debug_info.get("bar_row_end")
        if bar_rs is not None and bar_re is not None:
            cv2.rectangle(gray_bgr, (0, bar_rs), (w - 1, bar_re), (0, 255, 0), 1)

        # 画红段范围
        bar_start = debug_info.get("bar_start", 0)
        bar_end = debug_info.get("bar_end", 0)
        if bar_end > bar_start:
            # 在条行中间画
            mid_row = (bar_rs + bar_re) // 2 if bar_rs and bar_re else h // 2
            cv2.line(gray_bgr, (bar_start, mid_row), (bar_end, mid_row), (0, 0, 255), 2)

        # 红色掩码叠加（半透明红）
        red_mask = debug_info.get("red_mask")
        if red_mask is not None:
            overlay = np.zeros_like(gray_bgr)
            overlay[red_mask] = (0, 0, 200)
            gray_bgr = cv2.addWeighted(gray_bgr, 0.7, overlay, 0.3, 0)

        raw = debug_info.get("raw_value", -1)
        label = f"Eny:{raw:.0%}" if raw >= 0 else "Eny:-"
        cv2.putText(gray_bgr, label, (2, h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        return gray_bgr
