"""规划AI：OWLv2 识怪 + 特征检测 → 决策 → 下发指令。

状态机：
  EXPLORE ─→ OWLv2 检测到敌人 ─→ APPROACH ─→ 锁定成功 ─→ COMBAT
     ↑                                  │                         │
     └──────── 敌人消失/死亡 ←──────────┘←──────── 锁定丢失 ←─────┘
     ↑
  STUCK (撞墙脱困)

输入：单帧画面
输出：(mode, enemy_info)
  mode: "explore"/"approach"/"combat"/"stuck"
  enemy_info: None 或 {"side": "left"/"center"/"right", "count": N, "types": [...]}

用法：
    from planner import Planner
    planner = Planner()
    mode, enemy = planner.update(frame)
"""

import os
# OWLv2 模型用国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# 检测配置
DETECT_EVERY = 15       # 每 N 帧检测一次
CONFIDENCE = 0.08       # 置信度阈值（更低，Boss 战也能抓到）
QUERIES = ["enemy monster", "giant creature", "boss"]


class Planner:
    """画面特征 + OWLv2 零样本敌人检测 → 下发模式指令。"""

    def __init__(self):
        self.mode = "explore"

        # 状态机防抖
        self._mode_counter = 0
        self._mode_threshold = 5

        # 撞墙检测
        self._last_positions = []
        self._frame_count = 0

        # 画面尺寸
        self._h = 0
        self._w = 0

        # OWLv2 检测缓存
        self._owl_model = None
        self._owl_processor = None
        self._cached_enemies = None  # 最近一次检测结果

        # 延迟加载 OWLv2（首次 update 时加载，避免 import 时卡住）
        self._owl_loaded = False

    def _load_owl(self):
        """延迟加载 OWLv2（首次调用时加载模型）。"""
        print("  加载 OWLv2 视觉模型...")
        self._owl_model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16,
        ).to("cuda")
        self._owl_processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble")
        self._owl_model.eval()
        self._owl_loaded = True
        print("  OWLv2 就绪！")

    def update(self, frame) -> tuple:
        """返回 (mode, enemy_info)。"""
        self._frame_count += 1
        if self._h == 0:
            self._h, self._w = frame.shape[:2]

        # ── 特征检测 ──
        locked = self._detect_lock_circle(frame)
        stuck = self._detect_stuck(frame)
        enemies = self._detect_enemies_owl(frame)  # OWLv2，带缓存

        # ── 锁定确认（需要连续多帧，防假阳性）──
        if not hasattr(self, '_lock_cnt'):
            self._lock_cnt = 0
        # 锁定确认(3帧) + 丢失防抖(5帧) = 锁了就不轻易脱
        self._lock_cnt = self._lock_cnt + 1 if locked else max(0, self._lock_cnt - 1)
        lock_confirmed = self._lock_cnt >= 3
        lock_lost = self._lock_cnt <= 0  # 连续多帧无锁圈才算真丢了

        # ── 状态转移 ──
        target = self.mode
        if stuck:
            target = "stuck"
        elif lock_confirmed:
            target = "combat"   # 只有确认锁定后才战斗
        elif enemies is not None and enemies["count"] > 0:
            target = "approach"  # 有敌人未锁定 → 接近
        elif self.mode == "combat" and lock_lost and (enemies is None or enemies["count"] == 0):
            target = "explore"
        elif self.mode == "approach" and not locked and (enemies is None or enemies["count"] == 0):
            target = "explore"
        elif self.mode == "stuck" and not stuck:
            target = "explore"
        else:
            target = "explore"

        # 防抖
        if target != self.mode:
            self._mode_counter += 1
            if self._mode_counter >= self._mode_threshold:
                self._mode_counter = 0
                self.mode = target
        else:
            self._mode_counter = 0

        return self.mode, enemies

    def _detect_enemies_owl(self, frame):
        """OWLv2 零样本检测，每 N 帧跑一次，其余用缓存。"""
        if not self._owl_loaded:
            self._load_owl()

        if self._frame_count % DETECT_EVERY != 0:
            return self._cached_enemies

        # 降采样加速
        h, w = self._h, self._w
        small = cv2.resize(frame, (w // 2, h // 2))
        pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        inputs = self._owl_processor(text=QUERIES, images=pil, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._owl_model(**inputs)
        results = self._owl_processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=torch.tensor([(h // 2, w // 2)]), threshold=CONFIDENCE,
        )[0]

        boxes = results["boxes"].tolist()
        labels = results["labels"]
        scores = results["scores"].tolist()

        if len(boxes) == 0:
            self._cached_enemies = None
            return None

        # 分析敌人位置
        w2 = w // 2
        sides = []
        types = []
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 * 2  # 还原坐标（之前降采样了）
            if cx < w * 0.35:
                sides.append("left")
            elif cx > w * 0.65:
                sides.append("right")
            else:
                sides.append("center")
            types.append(QUERIES[label])

        # 多数敌人的位置
        from collections import Counter
        side = Counter(sides).most_common(1)[0][0]
        enemy_types = list(set(types))

        self._cached_enemies = {
            "side": side, "count": len(boxes),
            "types": enemy_types, "conf": max(scores),
        }
        return self._cached_enemies

    def _detect_lock_circle(self, frame) -> bool:
        h, w = self._h, self._w
        cx, cy = w // 2, h // 2
        rw, rh = w // 3, h // 3
        roi = frame[cy - rh // 2:cy + rh // 2, cx - rw // 2:cx + rw // 2]
        white = cv2.inRange(roi, np.array([150, 150, 150]), np.array([255, 255, 255]))
        kernel = np.ones((2, 2), np.uint8)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 25:  # 更小也认（远处敌人锁圈小）
                return True
        return False

    def _detect_stuck(self, frame) -> bool:
        if self._frame_count % 5 != 0:
            return self.mode == "stuck"
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self._last_positions.append(gray)
        if len(self._last_positions) < 10:
            return False
        if len(self._last_positions) > 30:
            self._last_positions.pop(0)
        diffs = []
        for i in range(1, len(self._last_positions)):
            diffs.append(np.abs(self._last_positions[i] - self._last_positions[i - 1]).mean())
        return np.mean(diffs[-10:]) < 0.5
