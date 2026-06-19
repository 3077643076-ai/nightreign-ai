"""检测器基类：EMA 平滑 + 自动校准 + 调试输出。"""

import cv2
import numpy as np


class BarDetector:
    """血条检测器基类。

    子类需要实现 detect(roi) → (raw_value, debug_dict)。
    基类负责 ROI 提取、EMA 平滑、自动校准。
    """

    def __init__(self, name, roi_fractions, ema_alpha=0.80):
        """
        Args:
            name: 检测器名称（hp/fp/stamina/lock/boss_hp/enemy_hp）
            roi_fractions: (y1_frac, y2_frac, x1_frac, x2_frac) 比例坐标
            ema_alpha: EMA 平滑系数（越大越平滑）
        """
        self.name = name
        self.roi_fractions = roi_fractions
        self.ema_alpha = ema_alpha
        self._ema_value = -1.0
        # 自动校准：记录运行中观察到的最大填充宽度
        self._max_fill_width = 0
        self._container_bounds = None  # (left, right) 容器边界缓存

    # ── ROI 提取 ──

    def extract_roi(self, frame):
        """从帧中提取 ROI 区域（BGR）。"""
        h, w = frame.shape[:2]
        y1 = int(self.roi_fractions[0] * h)
        y2 = int(self.roi_fractions[1] * h)
        x1 = int(self.roi_fractions[2] * w)
        x2 = int(self.roi_fractions[3] * w)
        return frame[y1:y2, x1:x2]

    # ── EMA 平滑 ──

    def smooth(self, raw_value):
        """EMA 平滑，返回平滑后的值。负数表示无效读数，不更新 EMA。"""
        if raw_value < 0:
            return self._ema_value if self._ema_value >= 0 else -1.0
        if self._ema_value < 0:
            self._ema_value = raw_value
        else:
            self._ema_value = (self.ema_alpha * self._ema_value
                               + (1 - self.ema_alpha) * raw_value)
        return self._ema_value

    # ── 容器校准（亮度法找血条底框） ──

    def calibrate_container(self, roi_bgr):
        """在 ROI 内找血条底框的左右边界。

        使用亮度法：血条底框比暗背景亮，找最长连续"亮段"。
        返回 (left, right) 列索引，失败返回 None。
        """
        h, w = roi_bgr.shape[:2]
        if h < 3 or w < 30:
            return None

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        col_max = gray.max(axis=0)  # 每列最大灰度

        # 暗背景参考（最暗 20% 列）
        sorted_max = np.sort(col_max)
        bg_cut = max(1, int(len(sorted_max) * 0.2))
        bg_level = float(sorted_max[:bg_cut].mean())

        # "有 UI" 的列 = 灰度明显高于暗背景
        has_ui = col_max > bg_level + 15

        # 找最长连续 UI 段
        best_left, best_right = 0, 0
        seg_start = -1
        for i in range(len(has_ui)):
            if has_ui[i]:
                if seg_start < 0:
                    seg_start = i
            else:
                if seg_start >= 0:
                    if i - seg_start > best_right - best_left:
                        best_left, best_right = seg_start, i
                    seg_start = -1
        if seg_start >= 0 and len(has_ui) - seg_start > best_right - best_left:
            best_left, best_right = seg_start, len(has_ui)

        if best_right - best_left < 30:
            return None
        return best_left, best_right

    # ── 自动校准填充宽度 ──

    def update_max_fill(self, fill_width):
        """更新观察到的最大填充宽度（用于自动校准 100% 基准）。"""
        if fill_width > self._max_fill_width:
            self._max_fill_width = fill_width

    def get_max_fill(self):
        return self._max_fill_width

    # ── 子类接口 ──

    def detect(self, frame):
        """检测血条。返回 (smoothed_value, debug_dict)。"""
        raise NotImplementedError

    def get_debug_image(self, roi, debug_info):
        """生成调试图。子类可覆写。默认返回灰度 ROI。"""
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
