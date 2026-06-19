"""游戏状态检测：从画面中提取 HP/FP/体力/敌人HP 等数值。

核心思路（纯亮度法，不依赖颜色）：
  1. 在用户标定的 ROI 内找血条容器（灰色底框 = 比暗背景亮的连续区域）
  2. 容器内用亮度区分"填充"和"空余"（填充比空余亮）
  3. 填充率 = 亮列数 / 容器宽
  4. EMA 平滑 + 锁定防抖

用法：
    from game_state import GameState
    gs = GameState()
    state = gs.detect(frame)
"""

import cv2
import numpy as np


class GameState:
    """检测艾尔登法环 黑夜君临 的游戏状态。"""

    # ── 各血条的屏幕区域（比例坐标，手动标定 @ 2560x1440）──
    # 黑夜君临从上到下: HP → FP → 体力
    REGIONS = {
        "hp":        (0.0347, 0.0500, 0.0844, 0.3898),  # HP 红条 最上
        "fp":        (0.0528, 0.0639, 0.0844, 0.3586),  # FP 蓝条 中间
        "stamina":   (0.0681, 0.0819, 0.0852, 0.4086),  # 体力绿条 最下
        "boss_hp":   (0.880, 0.950, 0.180, 0.870),
        "enemy_hp":  (0.075, 0.200, 0.100, 0.900),
        "lock_circle": (0.250, 0.800, 0.200, 0.800),
    }

    # EMA 平滑系数
    EMA_ALPHA = 0.85
    # 锁定防抖帧数
    LOCK_DEBOUNCE = 3

    def __init__(self, resolution=(1280, 720)):
        self._w, self._h = resolution

        # 容器边界缓存：{name: (left, right)} 在 ROI 内的列索引
        self._container = {}

        # EMA 缓存
        self._ema = {}

        # 锁定防抖
        self._lock_history = []
        self._locked = False

        # 血量差
        self._prev_hp = None
        self._prev_boss_hp = None

    # ================================================================
    # 工具
    # ================================================================

    def _get_roi(self, frame, region_name):
        y1_pct, y2_pct, x1_pct, x2_pct = self.REGIONS[region_name]
        h, w = frame.shape[:2]
        y1 = int(y1_pct * h)
        y2 = int(y2_pct * h)
        x1 = int(x1_pct * w)
        x2 = int(x2_pct * w)
        return frame[y1:y2, x1:x2], (y1, y2, x1, x2)

    # ================================================================
    # 血条检测（纯亮度法）
    # ================================================================

    def _measure_bar(self, roi_bgr):
        """纯亮度法测血条填充率。

        1. 在 ROI 内找灰色底框：亮度介于暗背景和填充之间的连续宽段
        2. 在底框内找"亮段"（填充）vs"暗段"（空余）的分界
        3. 填充率 = 亮段宽度 / 底框宽度

        返回 0.0~1.0，失败返回 -1.0。
        """
        h, w = roi_bgr.shape[:2]
        if h < 3 or w < 30:
            return -1.0

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # ── 第 1 步：找底框容器 ──
        # 每列最大灰度（能发现细血条）
        col_max = gray.max(axis=0)

        # 暗背景参考值（最暗 20% 列）
        sorted_max = np.sort(col_max)
        bg_cut = max(1, int(len(sorted_max) * 0.2))
        bg_level = float(sorted_max[:bg_cut].mean())

        # "有 UI"的列 = 灰度明显高于暗背景
        has_ui = col_max > bg_level + 20

        # 找最长的连续 UI 段 = 血条底框
        segments = []
        seg_start = -1
        for i in range(len(has_ui)):
            if has_ui[i]:
                if seg_start < 0:
                    seg_start = i
            else:
                if seg_start >= 0:
                    if i - seg_start >= 30:
                        segments.append((seg_start, i))
                    seg_start = -1
        if seg_start >= 0 and len(has_ui) - seg_start >= 30:
            segments.append((seg_start, len(has_ui)))

        if not segments:
            return -1.0

        # 取最宽的段作为底框
        bar_left, bar_right = max(segments, key=lambda s: s[1] - s[0])
        bar_width = bar_right - bar_left
        if bar_width < 30:
            return -1.0

        # ── 第 2 步：在底框内区分填充 vs 空余 ──
        # 取容器内的灰度，平滑后找亮度分界
        bar_gray = gray[:, bar_left:bar_right + 1]
        col_mean = bar_gray.mean(axis=0)  # 每列平均灰度

        # 平滑（减少噪声）
        kernel = np.ones(5) / 5
        col_smooth = np.convolve(col_mean, kernel, mode='same')

        # 找容器内的亮度范围
        bar_min = col_smooth.min()
        bar_max = col_smooth.max()

        if bar_max - bar_min < 15:
            # 对比度太低，可能没有填充（全空或全满）
            # 如果整体偏亮 → 可能全满；整体偏暗 → 可能全空
            if bar_max > 100:
                return 1.0  # 全亮 = 满血
            else:
                return 0.0  # 全暗 = 空血

        # Otsu 阈值把容器内分成"亮"和"暗"两类
        # 亮 = 填充，暗 = 空余
        bar_uint8 = ((col_smooth - bar_min) / max(1, bar_max - bar_min) * 255).astype(np.uint8)
        otsu_thresh, _ = cv2.threshold(
            bar_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_thresh = otsu_thresh / 255.0 * (bar_max - bar_min) + bar_min

        # 找到从"亮"到"暗"的过渡点 = 填充右边界
        # 从左边开始，亮→暗的第一次大幅下降
        fill_right = bar_width - 1
        for i in range(1, len(col_smooth)):
            if col_smooth[i - 1] > otsu_thresh and col_smooth[i] < otsu_thresh:
                fill_right = i - 1
                break
            if col_smooth[i] < otsu_thresh * 0.8 and col_smooth[max(0, i - 2)] > otsu_thresh:
                fill_right = i - 1
                break

        # 确认：如果最右边大部分都暗，说明有明确的空余部分
        right_third = col_smooth[-max(3, bar_width // 5):]
        if right_third.mean() < otsu_thresh - 5:
            # 有明确的空余，找到从亮到暗的过渡
            for i in range(bar_width - 1, 0, -1):
                if col_smooth[i] > otsu_thresh:
                    fill_right = i
                    break
        else:
            # 整体都亮 → 满血
            fill_right = bar_width - 1

        fill_width = fill_right + 1
        ratio = fill_width / bar_width

        if ratio < 0.005:
            return 0.0
        if ratio > 0.95:
            return 1.0
        return ratio

    # ================================================================
    # 锁定检测
    # ================================================================

    def _find_lock_circle(self, frame) -> tuple:
        """检测锁定光点。返回 (locked: bool, circle_y: int)。"""
        roi, (y1, y2, x1, x2) = self._get_roi(frame, "lock_circle")
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        _, bright = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_y = -1
        best_score = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3 or area > 60:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(1, bh)
            if not (0.7 < aspect < 1.4):
                continue
            dot_roi = gray[y:y+bh, x:x+bw]
            if dot_roi.max() < 235:
                continue
            cx = x + bw // 2
            cy = y + bh // 2
            r = max(bw, bh) // 2 + 3
            y1_r = max(0, cy - r)
            y2_r = min(gray.shape[0], cy + r)
            x1_r = max(0, cx - r)
            x2_r = min(gray.shape[1], cx + r)
            ring = gray[y1_r:y2_r, x1_r:x2_r]
            if ring.mean() < 110:
                score = area
                if score > best_score:
                    best_score = score
                    best_y = y1 + cy
        return best_score > 0, best_y

    def _update_lock(self, raw_locked: bool) -> bool:
        """防抖：连续 LOCK_DEBOUNCE 帧状态一致才切换。"""
        self._lock_history.append(raw_locked)
        if len(self._lock_history) > self.LOCK_DEBOUNCE:
            self._lock_history.pop(0)
        if len(self._lock_history) >= self.LOCK_DEBOUNCE:
            if all(self._lock_history):
                self._locked = True
            elif not any(self._lock_history):
                self._locked = False
        return self._locked

    # ================================================================
    # EMA 平滑
    # ================================================================

    def _smooth(self, name: str, raw_val: float) -> float:
        """EMA 平滑。"""
        if raw_val < 0:
            return self._ema.get(name, -1.0)
        if name not in self._ema or self._ema[name] < 0:
            self._ema[name] = raw_val
        else:
            self._ema[name] = (self.EMA_ALPHA * self._ema[name]
                               + (1 - self.EMA_ALPHA) * raw_val)
        return self._ema[name]

    # ================================================================
    # Boss 血条
    # ================================================================

    def _detect_boss_hp(self, frame) -> float:
        """Boss 底部血条：R/(G+B) 比值法。"""
        h, w = frame.shape[:2]
        y1, y2 = int(0.935 * h), h
        x1, x2 = 100, w - 100
        if y2 <= y1 or x2 <= x1:
            return -1.0
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return -1.0

        r = roi[:, :, 2].astype(float)
        g = roi[:, :, 1].astype(float)
        b = roi[:, :, 0].astype(float)
        ratio = r.mean(axis=0) / (g.mean(axis=0) + b.mean(axis=0) + 0.01)

        mean_ratio = ratio.mean()
        if mean_ratio < 0.62:
            return -1.0

        thresh = mean_ratio + 0.04
        hp_mask = ratio > thresh

        best_start, best_end = 0, 0
        cur_start = -1
        for i, val in enumerate(hp_mask):
            if val:
                if cur_start < 0: cur_start = i
            else:
                if cur_start >= 0:
                    if i - cur_start > best_end - best_start:
                        best_start, best_end = cur_start, i
                    cur_start = -1
        if cur_start >= 0:
            if len(hp_mask) - cur_start > best_end - best_start:
                best_start, best_end = cur_start, len(hp_mask)

        bar_width = best_end - best_start
        if bar_width < 50:
            return -1.0
        return bar_width / len(hp_mask)

    # ================================================================
    # 主入口
    # ================================================================

    def detect(self, frame) -> dict:
        """检测所有游戏状态。"""
        result = {
            "hp": -1.0, "fp": -1.0, "stamina": -1.0,
            "boss_hp": -1.0, "enemy_hp": -1.0,
            "locked": False, "hp_delta": 0.0, "boss_hp_delta": 0.0,
        }

        # ── 三条血条：亮度法 ──
        for name in ["hp", "fp", "stamina"]:
            roi, _ = self._get_roi(frame, name)
            raw_val = self._measure_bar(roi)
            smooth_val = self._smooth(name, raw_val)
            result[name] = smooth_val

        # ── HP 变化量 ──
        if result["hp"] >= 0 and self._prev_hp is not None and self._prev_hp >= 0:
            result["hp_delta"] = result["hp"] - self._prev_hp
        if result["hp"] >= 0:
            self._prev_hp = result["hp"]

        # ── 锁定 ──
        raw_locked, _ = self._find_lock_circle(frame)
        result["locked"] = self._update_lock(raw_locked)

        # ── Boss ──
        boss_val = self._detect_boss_hp(frame)
        if boss_val >= 0:
            result["boss_hp"] = self._smooth("boss_hp", boss_val)
            if self._prev_boss_hp is not None and self._prev_boss_hp >= 0:
                result["boss_hp_delta"] = result["boss_hp"] - self._prev_boss_hp
            self._prev_boss_hp = result["boss_hp"]

        # ── 小怪血条 ──
        if result["locked"]:
            enemy_roi, _ = self._get_roi(frame, "enemy_hp")
            if enemy_roi.size > 0:
                enemy_val = self._measure_bar(enemy_roi)
                if enemy_val >= 0:
                    result["enemy_hp"] = self._smooth("enemy_hp", enemy_val)

        return result

    @property
    def prev_hp(self):
        return self._prev_hp

    @property
    def prev_boss_hp(self):
        return self._prev_boss_hp


def draw_debug(frame, state: dict) -> np.ndarray:
    """在画面上绘制检测可视化。"""
    img = frame.copy()
    h, w = img.shape[:2]
    gs = GameState((w, h))

    def _draw_roi(name, color, value):
        r = gs.REGIONS[name]
        x1, y1 = int(r[2] * w), int(r[0] * h)
        x2, y2 = int(r[3] * w), int(r[1] * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        label = f'{name}: {value:.0%}' if value >= 0 else f'{name}: -'
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    _draw_roi("hp", (0, 200, 0), state.get("hp", -1))
    _draw_roi("fp", (200, 150, 0), state.get("fp", -1))
    _draw_roi("stamina", (0, 200, 200), state.get("stamina", -1))
    _draw_roi("boss_hp", (0, 0, 255), state.get("boss_hp", -1))
    _draw_roi("enemy_hp", (0, 0, 255), state.get("enemy_hp", -1))

    r = gs.REGIONS["lock_circle"]
    x1, y1 = int(r[2] * w), int(r[0] * h)
    x2, y2 = int(r[3] * w), int(r[1] * h)
    lock_color = (0, 255, 0) if state.get("locked") else (100, 100, 100)
    cv2.rectangle(img, (x1, y1), (x2, y2), lock_color, 1)
    cv2.putText(img, f'LOCK: {state.get("locked", False)}', (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, lock_color, 1)

    parts = []
    for k in ["hp", "fp", "stamina", "boss_hp", "enemy_hp"]:
        v = state.get(k, -1)
        if v >= 0:
            parts.append(f'{k}={v:.0%}')
    cv2.putText(img, " | ".join(parts), (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return img


def compute_reward(prev_state: dict, curr_state: dict) -> float:
    """根据状态变化计算奖励值。"""
    reward = 0.0
    hp_delta = curr_state.get("hp_delta", 0.0)
    if hp_delta < -0.02:
        reward += hp_delta * 250
    elif hp_delta > 0.02:
        reward += 3.0
    boss_delta = curr_state.get("boss_hp_delta", 0.0)
    if boss_delta < -0.01:
        reward += abs(boss_delta) * 500
    if curr_state.get("hp", 1.0) <= 0.05 and prev_state.get("hp", 1.0) > 0.05:
        reward -= 10.0
    if curr_state.get("boss_hp", -1) <= 0.02 and prev_state.get("boss_hp", -1) > 0.02:
        reward += 50.0
    return reward
