"""体力绿条检测器 — 颜色比值 + 亮度混合法。

绿色信号比 FP 强但不如 HP，单独 G/(R+G+B)>0.36 会低估。
混合策略：绿色比值给大致范围，亮度 Otsu 精确定位填充边界。
体力% = 亮区宽度 / 容器总宽度（用亮度微调后的边界）。
"""

import cv2
import numpy as np
from perception.base import BarDetector


class StaminaDetector(BarDetector):
    """体力绿条检测器。

    ROI: 比例坐标 y=0.0681~0.0819, x=0.0852~0.4086 (FP 下方)
    """

    # 绿色比值阈值
    GREEN_RATIO_THRESHOLD = 0.35
    # 容器内最小对比度
    MIN_CONTRAST = 10

    def __init__(self):
        super().__init__(
            name="stamina",
            roi_fractions=(0.0681, 0.0819, 0.0852, 0.4086),
            ema_alpha=0.78,  # 体力变化快，用更小的平滑系数
        )

    def detect(self, frame):
        """检测体力百分比（混合法）。

        1. 绿色比值法初筛找到大致填充范围
        2. 亮度 Otsu 精确定位亮→暗过渡
        3. 综合两者确定体力%
        """
        roi = self.extract_roi(frame)
        if roi.size == 0:
            return -1.0, {}

        h, w = roi.shape[:2]

        # ── 颜色比值法：绿色掩码 ──
        b = roi[:, :, 0].astype(np.float32)
        g = roi[:, :, 1].astype(np.float32)
        r = roi[:, :, 2].astype(np.float32)
        total = r + g + b + 1e-6
        green_ratio = g / total
        green_mask = green_ratio > self.GREEN_RATIO_THRESHOLD

        col_green_count = green_mask.sum(axis=0)
        min_green_per_col = max(1, int(h * 0.25))
        col_green_filled = col_green_count >= min_green_per_col

        # 找绿色填充右边界（允许小间断）
        green_fill_right = 0
        gap = 0
        for i in range(w):
            if col_green_filled[i]:
                green_fill_right = i
                gap = 0
            else:
                gap += 1
                if gap > 3:
                    break

        # ── 亮度法：容器内亮暗分离 ──
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        container = self.calibrate_container(roi)
        if container is None:
            bar_left, bar_right = 0, w - 1
        else:
            bar_left, bar_right = container

        bar_width = bar_right - bar_left + 1
        if bar_width < 20:
            return -1.0, {
                "roi": roi,
                "green_mask": green_mask,
                "green_fill_right": green_fill_right,
            }

        bar_gray = gray[:, bar_left:bar_right + 1]
        col_mean = bar_gray.mean(axis=0)
        kernel = np.ones(3) / 3
        col_smooth = np.convolve(col_mean, kernel, mode='same')

        bar_min = col_smooth.min()
        bar_max = col_smooth.max()

        bright_fill_right = bar_width - 1  # 默认全满
        if bar_max - bar_min >= self.MIN_CONTRAST:
            bar_norm = ((col_smooth - bar_min) / (bar_max - bar_min) * 255).astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(
                bar_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            actual_thresh = otsu_thresh / 255.0 * (bar_max - bar_min) + bar_min

            for i in range(1, bar_width):
                if col_smooth[i - 1] > actual_thresh and col_smooth[i] < actual_thresh:
                    bright_fill_right = i - 1
                    break

            # 验证
            right_portion = col_smooth[-max(2, bar_width // 5):]
            if right_portion.mean() > actual_thresh - 3:
                bright_fill_right = bar_width - 1

        # ── 混合：取两者中靠右的（更保守的估计） ──
        # 绿色比值法在低体力时较准，亮度法在整体都暗时可能误判
        # 策略：用绿色比值做个引导，亮度法做精确定位
        # 如果绿色法和亮度法结果接近，取亮度法；如果差很远，偏向亮度法
        green_fill_norm = green_fill_right / max(1, w)
        bright_fill_norm = bright_fill_right / max(1, bar_width)

        if abs(green_fill_norm - bright_fill_norm) < 0.25:
            # 接近，用亮度法
            final_fill_right = bright_fill_right
            container_w = bar_width
        else:
            # 差很远，亮度法更可信（绿色太弱）
            final_fill_right = bright_fill_right
            container_w = bar_width

        raw_value = final_fill_right / max(1, container_w)
        raw_value = max(0.0, min(1.0, raw_value))

        smooth_value = self.smooth(raw_value)

        debug = {
            "roi": roi,
            "green_mask": green_mask,
            "green_fill_right": green_fill_right,
            "bar_left": bar_left,
            "bar_right": bar_right,
            "col_smooth": col_smooth,
            "bright_fill_right": bright_fill_right,
            "raw_value": raw_value,
        }

        return smooth_value, debug

    def get_debug_image(self, roi, debug_info):
        """生成调试图像：灰度 ROI | 绿色掩码 | 亮度曲线。"""
        if roi is None or roi.size == 0:
            return np.zeros((40, 300), dtype=np.uint8)

        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        green_mask = debug_info.get("green_mask")
        mask_vis = np.zeros_like(gray)
        if green_mask is not None:
            mask_vis = (green_mask * 255).astype(np.uint8)

        plot_h = 32
        out = np.zeros((h + plot_h + 2, w * 2, 3), dtype=np.uint8)

        # 左半：灰度
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out[:h, :w] = gray_bgr

        bar_left = debug_info.get("bar_left", 0)
        bar_right = debug_info.get("bar_right", w - 1)
        cv2.line(out, (bar_left, 0), (bar_left, h - 1), (0, 255, 0), 1)
        cv2.line(out, (bar_right, 0), (bar_right, h - 1), (0, 255, 0), 1)
        # 绿色填充右边界
        green_fr = debug_info.get("green_fill_right", 0)
        cv2.line(out, (green_fr, 0), (green_fr, h - 1), (0, 200, 0), 1)

        # 右半：绿色掩码
        mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        out[:h, w:] = mask_bgr

        # 下方：亮度曲线
        col_smooth = debug_info.get("col_smooth")
        if col_smooth is not None and len(col_smooth) > 0:
            cv = col_smooth
            cv_min, cv_max = cv.min(), cv.max()
            cv_range = max(1, cv_max - cv_min)
            for x in range(min(w, len(cv))):
                y_norm = (cv[x] - cv_min) / cv_range
                py = h + 2 + int((1 - y_norm) * (plot_h - 2))
                if 0 <= py < out.shape[0]:
                    out[py, x] = (255, 255, 255)

        raw = debug_info.get("raw_value", -1)
        label = f"STM:{raw:.0%}" if raw >= 0 else "STM:-"
        cv2.putText(out, label, (2, h + plot_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        return out
