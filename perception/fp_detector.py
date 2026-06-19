"""FP 蓝条检测器 — 亮度梯度法。

FP 蓝色太弱（B≈G），颜色比值法完全无效。
改用亮度梯度法：填充区域比空余区域亮，找亮→暗过渡点。
FP% = 亮区宽度 / 容器总宽度。
"""

import cv2
import numpy as np
from perception.base import BarDetector


class FPDetector(BarDetector):
    """FP 蓝条检测器。

    ROI: 比例坐标 y=0.0528~0.0639, x=0.0844~0.3586 (HP 下方)
    """

    # 容器内亮暗分离的最小对比度
    MIN_CONTRAST = 12

    def __init__(self):
        super().__init__(
            name="fp",
            roi_fractions=(0.0528, 0.0639, 0.0844, 0.3586),
            ema_alpha=0.82,
        )

    def detect(self, frame):
        """检测 FP 百分比（亮度梯度法）。

        1. 找血条底框容器
        2. 容器内用 Otsu 阈值分开亮区（填充）和暗区（空余）
        3. 亮区宽度 / 容器宽度 = FP%
        """
        roi = self.extract_roi(frame)
        if roi.size == 0:
            return -1.0, {}

        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ── 第 1 步：找容器 ──
        container = self.calibrate_container(roi)
        if container is None:
            # 找不到容器，把整个 ROI 当容器
            bar_left, bar_right = 0, w - 1
        else:
            bar_left, bar_right = container

        bar_width = bar_right - bar_left + 1
        if bar_width < 20:
            return -1.0, {"roi": roi, "gray": gray, "bar_left": bar_left, "bar_right": bar_right}

        # ── 第 2 步：容器内灰度分析 ──
        bar_gray = gray[:, bar_left:bar_right + 1]
        col_mean = bar_gray.mean(axis=0)  # 每列平均灰度

        # 平滑降噪
        kernel = np.ones(3) / 3
        col_smooth = np.convolve(col_mean, kernel, mode='same')

        bar_min = col_smooth.min()
        bar_max = col_smooth.max()

        # 对比度太低 → 全满或全空
        if bar_max - bar_min < self.MIN_CONTRAST:
            if bar_max > 80:
                raw_value = 1.0
            else:
                raw_value = 0.0
        else:
            # Otsu 阈值分亮暗
            bar_norm = ((col_smooth - bar_min) / (bar_max - bar_min) * 255).astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(
                bar_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Otsu 阈值映射回原始灰度
            actual_thresh = otsu_thresh / 255.0 * (bar_max - bar_min) + bar_min

            # 亮区在左边（填充从左开始），找到亮→暗过渡
            fill_right = bar_width - 1  # 默认全亮 = 满
            for i in range(1, bar_width):
                # 亮→暗的过渡
                if col_smooth[i - 1] > actual_thresh and col_smooth[i] < actual_thresh:
                    fill_right = i - 1
                    break

            # 验证：右边区域是否真的暗
            right_portion = col_smooth[-max(2, bar_width // 5):]
            if right_portion.mean() > actual_thresh - 3:
                # 右边不暗 → 可能全满
                fill_right = bar_width - 1

            fill_width = fill_right + 1
            raw_value = fill_width / bar_width
            raw_value = max(0.0, min(1.0, raw_value))

        smooth_value = self.smooth(raw_value)

        debug = {
            "roi": roi,
            "gray": gray,
            "bar_left": bar_left,
            "bar_right": bar_right,
            "col_smooth": col_smooth,
            "bar_min": bar_min,
            "bar_max": bar_max,
            "raw_value": raw_value,
        }

        return smooth_value, debug

    def get_debug_image(self, roi, debug_info):
        """生成调试图像：上半灰度 ROI + 亮度曲线，下半画容器和分界。"""
        if roi is None or roi.size == 0:
            return np.zeros((40, 300), dtype=np.uint8)

        h, w = roi.shape[:2]
        gray = debug_info.get("gray")
        if gray is None:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        bar_left = debug_info.get("bar_left", 0)
        bar_right = debug_info.get("bar_right", w - 1)
        col_smooth = debug_info.get("col_smooth")
        raw_value = debug_info.get("raw_value", -1)

        # 创建输出图像：上方是灰度 ROI + 标记线，下方是亮度曲线图
        plot_h = 40
        out = np.zeros((h + plot_h + 2, w, 3), dtype=np.uint8)

        # 上半：灰度 ROI（转为 BGR 彩色以画彩色标记）
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out[:h, :, :] = gray_bgr

        # 画容器边界（绿线）
        cv2.line(out, (bar_left, 0), (bar_left, h - 1), (0, 255, 0), 1)
        cv2.line(out, (bar_right, 0), (bar_right, h - 1), (0, 255, 0), 1)

        # 下半：亮度曲线
        if col_smooth is not None:
            bar_w = bar_right - bar_left + 1
            # 将曲线缩放到 plot_h 高度
            curve_min = col_smooth.min()
            curve_max = col_smooth.max()
            curve_range = max(1, curve_max - curve_min)

            for x in range(min(w, len(col_smooth))):
                y_norm = (col_smooth[x] - curve_min) / curve_range
                py = h + 2 + int((1 - y_norm) * (plot_h - 2))
                if 0 <= py < out.shape[0]:
                    # 在容器内用白色，容器外用灰色
                    color = (255, 255, 255) if bar_left <= x <= bar_right else (80, 80, 80)
                    out[py, x] = color

            # 画 Otsu 阈值线
            bar_gray_vals = col_smooth[bar_left:bar_right + 1]
            if len(bar_gray_vals) > 0:
                bmin, bmax = bar_gray_vals.min(), bar_gray_vals.max()
                if bmax - bmin > self.MIN_CONTRAST:
                    bar_norm = ((bar_gray_vals - bmin) / (bmax - bmin) * 255).astype(np.uint8)
                    otsu_thresh, _ = cv2.threshold(
                        bar_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    actual_thresh = otsu_thresh / 255.0 * (bmax - bmin) + bmin
                    thresh_y = h + 2 + int((1 - (actual_thresh - curve_min) / curve_range) * (plot_h - 2))
                    cv2.line(out, (0, thresh_y), (w - 1, thresh_y), (0, 200, 200), 1)

        # 标注
        label = f"FP:{raw_value:.0%}" if raw_value >= 0 else "FP:-"
        cv2.putText(out, label, (2, h + plot_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        return out
