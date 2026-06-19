"""Boss 血条检测器 — 底部红色比值法。

Boss 血条在屏幕底部，很宽很长。
用 R/(G+B) 比值找红色区域，找最长连续红段作为血条。
Boss HP% = 红色段长度 / 总 ROI 宽度。
"""

import cv2
import numpy as np
from perception.base import BarDetector


class BossHPDetector(BarDetector):
    """Boss 血条检测器。

    ROI: 屏幕底部 y=0.88~0.95, x=0.18~0.87
    """

    # R/(G+B) 比值阈值
    RED_RATIO_THRESHOLD = 0.62
    # 最小血条宽度（像素）
    MIN_BAR_WIDTH = 50

    def __init__(self):
        super().__init__(
            name="boss_hp",
            roi_fractions=(0.88, 0.95, 0.18, 0.87),
            ema_alpha=0.85,
        )

    def detect(self, frame):
        """检测 Boss HP 百分比。

        用 R/(G+B) 比值法，找红色区域在 ROI 宽度中的占比。
        """
        roi = self.extract_roi(frame)
        if roi.size == 0:
            return -1.0, {}

        h, w = roi.shape[:2]

        # ── R/(G+B) 比值 ──
        r = roi[:, :, 2].astype(np.float32)
        g = roi[:, :, 1].astype(np.float32)
        b = roi[:, :, 0].astype(np.float32)
        # 沿列取平均，得到每列的 R/(G+B) 比值
        r_mean = r.mean(axis=0)
        g_mean = g.mean(axis=0)
        b_mean = b.mean(axis=0)
        ratio = r_mean / (g_mean + b_mean + 0.01)

        # ── 判断是否有 Boss 血条 ──
        mean_ratio = ratio.mean()
        if mean_ratio < self.RED_RATIO_THRESHOLD:
            # 没有明显的红色 → 无 Boss 血条
            return -1.0, {
                "roi": roi,
                "ratio": ratio,
                "mean_ratio": mean_ratio,
            }

        # ── 找红色区域 ──
        thresh = mean_ratio + 0.04
        hp_mask = ratio > thresh

        # 找最长连续红段
        best_start, best_end = 0, 0
        cur_start = -1
        for i, val in enumerate(hp_mask):
            if val:
                if cur_start < 0:
                    cur_start = i
            else:
                if cur_start >= 0:
                    if i - cur_start > best_end - best_start:
                        best_start, best_end = cur_start, i
                    cur_start = -1
        if cur_start >= 0:
            if len(hp_mask) - cur_start > best_end - best_start:
                best_start, best_end = cur_start, len(hp_mask)

        bar_width = best_end - best_start
        if bar_width < self.MIN_BAR_WIDTH:
            return -1.0, {
                "roi": roi,
                "ratio": ratio,
                "mean_ratio": mean_ratio,
                "bar_width": bar_width,
            }

        # Boss HP% = 红段宽度 / 总宽度
        raw_value = bar_width / w
        raw_value = max(0.0, min(1.0, raw_value))

        smooth_value = self.smooth(raw_value)

        debug = {
            "roi": roi,
            "ratio": ratio,
            "hp_mask": hp_mask,
            "best_start": best_start,
            "best_end": best_end,
            "bar_width": bar_width,
            "mean_ratio": mean_ratio,
            "raw_value": raw_value,
        }

        return smooth_value, debug

    def get_debug_image(self, roi, debug_info):
        """生成调试图像：灰度 ROI + 红色比值曲线 + 检测结果。"""
        if roi is None or roi.size == 0:
            return np.zeros((40, 400), dtype=np.uint8)

        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        ratio = debug_info.get("ratio")
        hp_mask = debug_info.get("hp_mask")
        best_start = debug_info.get("best_start", 0)
        best_end = debug_info.get("best_end", 0)

        plot_h = 40
        out = np.zeros((h + plot_h + 2, w, 3), dtype=np.uint8)

        # 上半：灰度 ROI
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out[:h, :, :] = gray_bgr

        # 画检测到的红段
        cv2.rectangle(out, (best_start, 0), (best_end, h - 1), (0, 0, 255), 1)

        # 下半：红色比值曲线
        if ratio is not None and len(ratio) > 0:
            r_min, r_max = ratio.min(), ratio.max()
            r_range = max(0.01, r_max - r_min)
            mean_ratio = debug_info.get("mean_ratio", 0.6)
            thresh = mean_ratio + 0.04

            for x in range(min(w, len(ratio))):
                y_norm = (ratio[x] - r_min) / r_range
                py = h + 2 + int((1 - y_norm) * (plot_h - 2))
                if 0 <= py < out.shape[0]:
                    is_hp = hp_mask[x] if hp_mask is not None else False
                    color = (0, 0, 255) if is_hp else (100, 100, 100)
                    out[py, x] = color

            # 阈值线
            thresh_y = h + 2 + int((1 - (thresh - r_min) / r_range) * (plot_h - 2))
            cv2.line(out, (0, thresh_y), (w - 1, thresh_y), (0, 200, 200), 1)

        raw = debug_info.get("raw_value", -1)
        label = f"Boss:{raw:.0%}" if raw >= 0 else "Boss:-"
        cv2.putText(out, label, (2, h + plot_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        return out
