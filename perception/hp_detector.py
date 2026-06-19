"""HP 红条检测器 — 红色比值法。

原理：R/(R+G+B) > 0.40 精准区分红色填充区域和暗色空余区域。
HP% = 当前红色填充宽度 / 自动校准的满血红色宽度。
"""

import cv2
import numpy as np
from perception.base import BarDetector


class HPDetector(BarDetector):
    """HP 红条检测器。

    ROI: 比例坐标 y=0.0347~0.0500, x=0.0844~0.3898 (屏幕左上角)
    """

    # 红色比值阈值：超过此值认为是红色填充像素
    RED_RATIO_THRESHOLD = 0.40
    # 每列最少红色像素占比（相对于 ROI 高度）
    MIN_RED_COL_FRAC = 0.25

    def __init__(self):
        super().__init__(
            name="hp",
            roi_fractions=(0.0347, 0.0500, 0.0844, 0.3898),
            ema_alpha=0.82,
        )

    def detect(self, frame):
        """检测 HP 百分比。

        Returns:
            smoothed_value: 0.0~1.0 的 HP 百分比，-1.0 表示检测失败
            debug_dict: 包含中间结果的字典
        """
        roi = self.extract_roi(frame)
        if roi.size == 0:
            return -1.0, {}

        h, w = roi.shape[:2]

        # ── 计算红色比值掩码 ──
        b = roi[:, :, 0].astype(np.float32)
        g = roi[:, :, 1].astype(np.float32)
        r = roi[:, :, 2].astype(np.float32)
        total = r + g + b + 1e-6
        red_ratio = r / total
        red_mask = red_ratio > self.RED_RATIO_THRESHOLD

        # ── 每列红色像素数 ──
        col_red_count = red_mask.sum(axis=0)  # shape: (w,)
        min_red_per_col = max(1, int(h * self.MIN_RED_COL_FRAC))
        col_filled = col_red_count >= min_red_per_col

        # ── 找红色填充的右边界 ──
        # 从左开始，允许小间断（1~2 像素），找到红色区域的右端
        fill_right = 0
        gap_count = 0
        for i in range(w):
            if col_filled[i]:
                fill_right = i
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > 2:  # 连续 3 列无红色 = 填充结束
                    break
                # 小间断：先继续，但不更新 fill_right

        # ── 自动校准满血基准 ──
        if fill_right > 0:
            self.update_max_fill(fill_right)

        max_fill = self.get_max_fill()
        if max_fill < 10:
            return -1.0, {
                "roi": roi,
                "red_mask": red_mask,
                "col_red_count": col_red_count,
                "fill_right": fill_right,
                "max_fill": max_fill,
            }

        hp_pct = fill_right / max_fill
        # 钳制到 [0, 1]
        hp_pct = max(0.0, min(1.0, hp_pct))

        raw_value = hp_pct
        smooth_value = self.smooth(raw_value)

        debug = {
            "roi": roi,
            "red_mask": red_mask,
            "col_red_count": col_red_count,
            "col_filled": col_filled,
            "fill_right": fill_right,
            "max_fill": max_fill,
            "raw_value": raw_value,
        }

        return smooth_value, debug

    def get_debug_image(self, roi, debug_info):
        """生成调试图像：左边灰度 ROI，右边红色掩码。"""
        if roi is None or roi.size == 0:
            return np.zeros((20, 200), dtype=np.uint8)

        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 红色掩码可视化为白色
        red_mask = debug_info.get("red_mask")
        if red_mask is None:
            return gray

        mask_vis = (red_mask * 255).astype(np.uint8)

        # 水平拼接：灰度 | 掩码
        combined = np.hstack([gray, mask_vis])

        # 画填充右边界线
        fill_right = debug_info.get("fill_right", 0)
        max_fill = debug_info.get("max_fill", w)
        if fill_right > 0:
            cv2.line(combined, (fill_right, 0), (fill_right, h - 1), 128, 1)
        if max_fill < w:
            cv2.line(combined, (max_fill, 0), (max_fill, h - 1), 64, 1)
        # 在掩码那半也画
        if fill_right > 0:
            cv2.line(combined, (w + fill_right, 0), (w + fill_right, h - 1), 128, 1)

        # 标注数值
        raw = debug_info.get("raw_value", -1)
        label = f"HP:{raw:.0%}" if raw >= 0 else "HP:-"
        cv2.putText(combined, label, (2, h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)

        return combined
