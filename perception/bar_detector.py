"""血条检测器 — 灰度峰值 + 颜色填充。

稳定措施:
  1. 边框位置取最近15帧中位数
  2. max_fill 只增不减 (满血时自动校准)
  3. 输出 EMA 0.92 (很平滑)
"""

import cv2
import numpy as np
from collections import deque


class PeakBarDetector:

    def __init__(self, name, roi_frac, color_channel="red",
                 ratio_threshold=0.38, default_max=150):
        self.name = name
        self.roi_frac = roi_frac
        self.color_channel = color_channel
        self.ratio_threshold = ratio_threshold
        self._max_fill = float(default_max)
        self._ema_val = -1.0
        self._border_hist = deque(maxlen=15)
        self._last_border = -1

    def extract_roi(self, frame):
        h, w = frame.shape[:2]
        y1, y2, x1, x2 = self.roi_frac
        return frame[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]

    def detect(self, frame):
        roi = self.extract_roi(frame)
        if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 20:
            return (self._ema_val if self._ema_val >= 0 else -1.0), {}

        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cmax = gray.max(axis=0).astype(float)

        # ── 1. 找边框: gray_max 最高峰 (>150) ──
        best_col = -1
        best_val = 0
        for i in range(5, w - 5):
            v = cmax[i]
            if v < 150:
                continue
            if v >= cmax[i-1] and v >= cmax[i+1] and \
               v >= cmax[i-3] and v >= cmax[i+3] and \
               (v > cmax[i-1] or v > cmax[i+1]):
                if v > best_val:
                    best_val = v
                    best_col = i
                    while best_col + 1 < w and abs(cmax[best_col+1] - v) < 2:
                        best_col += 1

        if best_col < 0:
            return (self._ema_val if self._ema_val >= 0 else -1.0), {}

        # ── 中位数去抖 ──
        self._border_hist.append(best_col)
        border = int(np.median(list(self._border_hist)))
        self._last_border = border

        # ── 2. 左边缘 ──
        bg = float(np.sort(cmax)[:max(1, int(w * 0.15))].mean())
        left = 0
        for i in range(border, -1, -1):
            if cmax[i] < bg + 15:
                left = i + 1
                break

        if border - left < 12:
            return (self._ema_val if self._ema_val >= 0 else -1.0), {}

        # ── 3. 填充终点 ──
        if self.color_channel == "border_only":
            fill_end = border
        else:
            fill_end = self._find_fill(roi, left, border)

        fill_w = fill_end - left + 1

        # ── 4. 校准: 填充贴近边框时 = 满血，更新基准 ──
        bar_w = border - left + 1
        fill_ratio_of_bar = fill_w / max(1, bar_w)
        # 填充超过边框的90% → 视作满血
        if fill_ratio_of_bar > 0.88:
            self._max_fill = self._max_fill * 0.7 + bar_w * 0.3
        # 如果当前填充比 max 大很多 → 快速学习
        elif fill_w > self._max_fill * 1.1:
            self._max_fill = fill_w
        self._max_fill = max(20, self._max_fill)

        raw = fill_w / self._max_fill
        raw = max(0.0, min(1.0, raw))

        # ── 5. EMA 0.92 重平滑 ──
        if self._ema_val < 0:
            self._ema_val = raw
        else:
            self._ema_val = self._ema_val * 0.92 + raw * 0.08

        return self._ema_val, {
            "border_col": border, "left_col": left,
            "fill_end": fill_end, "fill_width": fill_w,
            "raw": raw,
        }

    def _find_fill(self, roi, left, border):
        h = roi.shape[0]
        container = roi[:, left:border+1]
        ch, cw = container.shape[:2]
        if ch < 2 or cw < 3:
            return border

        b = container[:,:,0].astype(np.float32)
        g = container[:,:,1].astype(np.float32)
        r = container[:,:,2].astype(np.float32)
        total = r + g + b + 1e-6
        ratio = (r / total) if self.color_channel == "red" else (g / total)
        col_frac = (ratio > self.ratio_threshold).sum(axis=0) / ch

        consecutive = 0
        end = left
        for i in range(cw - 1, -1, -1):
            if col_frac[i] >= 0.30:
                if consecutive == 0:
                    end = left + i
                consecutive += 1
            else:
                if consecutive >= 3:
                    break
                consecutive = 0
        return end
