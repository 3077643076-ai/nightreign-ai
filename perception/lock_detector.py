"""锁定圈检测器 — 找锁定光点。

原理：锁定状态下屏幕中间出现一个白色亮点，周围有暗圈。
检测白色亮点 + 周围暗环来确认锁定状态。
"""

import cv2
import numpy as np
from perception.base import BarDetector


class LockDetector(BarDetector):
    """锁定圈检测器。

    ROI: 屏幕中央区域 y=0.25~0.80, x=0.20~0.80
    """

    # 白色亮点阈值
    BRIGHT_THRESHOLD = 220
    # 暗环灰度上限
    RING_DARK_MAX = 130
    # 防抖帧数
    DEBOUNCE_FRAMES = 3

    def __init__(self):
        super().__init__(
            name="lock",
            roi_fractions=(0.25, 0.80, 0.20, 0.80),
            ema_alpha=0.90,
        )
        self._lock_history = []
        self._locked = False

    def detect(self, frame):
        """检测锁定状态。

        Returns:
            smooth_value: 1.0 = 已锁定, 0.0 = 未锁定, -1.0 = 无法判断
            debug_dict
        """
        roi = self.extract_roi(frame)
        if roi.size == 0:
            return -1.0 if not self._locked else 1.0, {}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ── 找高亮像素（潜在锁定光点） ──
        _, bright = cv2.threshold(gray, self.BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score = 0
        locked = False
        lock_y = -1
        all_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 锁定光点通常很小（3~60 像素）
            if area < 3 or area > 60:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(1, bh)
            # 接近圆形（宽高比 0.7~1.4）
            if not (0.6 < aspect < 1.6):
                continue

            # 亮点内部灰度要够高
            dot_roi = gray[y:y + bh, x:x + bw]
            if dot_roi.max() < 230:
                continue

            # 检查周围暗环
            cx = x + bw // 2
            cy = y + bh // 2
            r = max(bw, bh) // 2 + 3
            ring_y1 = max(0, cy - r)
            ring_y2 = min(gray.shape[0], cy + r)
            ring_x1 = max(0, cx - r)
            ring_x2 = min(gray.shape[1], cx + r)
            ring = gray[ring_y1:ring_y2, ring_x1:ring_x2]

            ring_mean = ring.mean()
            all_candidates.append({"area": area, "cx": cx, "cy": cy, "ring_mean": ring_mean})

            # 周围暗环的灰度要够低（锁定圈的特点）
            if ring_mean < self.RING_DARK_MAX:
                score = area
                if score > best_score:
                    best_score = score
                    lock_y = cy
                    locked = True

        # ── 防抖 ──
        self._lock_history.append(locked)
        if len(self._lock_history) > self.DEBOUNCE_FRAMES:
            self._lock_history.pop(0)

        if len(self._lock_history) >= self.DEBOUNCE_FRAMES:
            if all(self._lock_history):
                self._locked = True
            elif not any(self._lock_history):
                self._locked = False

        raw_value = 1.0 if self._locked else 0.0

        debug = {
            "roi": roi,
            "gray": gray,
            "bright": bright,
            "candidates": all_candidates,
            "locked": self._locked,
            "lock_y": lock_y,
        }

        return raw_value, debug

    def is_locked(self):
        return self._locked

    def get_debug_image(self, roi, debug_info):
        """生成调试图像：灰度 ROI 叠加高亮像素。"""
        if roi is None or roi.size == 0:
            return np.zeros((100, 200), dtype=np.uint8)

        gray = debug_info.get("gray")
        if gray is None:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 转 BGR 以画彩色标记
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 标记所有候选光点
        for cand in debug_info.get("candidates", []):
            cx, cy = cand["cx"], cand["cy"]
            area = cand["area"]
            ring = cand["ring_mean"]
            # 绿圈 = 通过暗环检测，红圈 = 未通过
            color = (0, 255, 0) if ring < self.RING_DARK_MAX else (0, 0, 255)
            cv2.circle(out, (cx, cy), 8, color, 1)
            txt = f"r={ring:.0f}"
            cv2.putText(out, txt, (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # 状态标注
        locked = debug_info.get("locked", False)
        status = "LOCKED" if locked else "no lock"
        color = (0, 255, 0) if locked else (0, 0, 255)
        cv2.putText(out, status, (4, out.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return out
