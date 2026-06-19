"""perception — 游戏 HUD 状态检测（特征峰算法）。

用法:
    from perception import GameState
    gs = GameState()
    state = gs.detect(frame)  # {"hp": 0.79, "fp": 0.96, "stamina": 0.14, ...}
"""

from perception.bar_detector import PeakBarDetector
from perception.lock_detector import LockDetector
from perception.boss_hp_detector import BossHPDetector
from perception.enemy_hp_detector import EnemyHPDetector


class GameState:
    """游戏状态检测器 — 统一入口。"""

    # ROI 比例坐标 (已标定 @ 1280x720)
    ROI = {
        "hp":      (0.0347, 0.0500, 0.0844, 0.3898),
        "fp":      (0.0528, 0.0639, 0.0844, 0.3586),
        "stamina": (0.0681, 0.0819, 0.0852, 0.4086),
    }

    def __init__(self, resolution=(1280, 720)):
        self._w, self._h = resolution

        self._detectors = {
            "hp": PeakBarDetector("hp", self.ROI["hp"],
                                  color_channel="red", ratio_threshold=0.38,
                                  default_max=156),
            "fp": PeakBarDetector("fp", self.ROI["fp"],
                                  color_channel="border_only",
                                  default_max=106),
            "stamina": PeakBarDetector("stamina", self.ROI["stamina"],
                                       color_channel="green", ratio_threshold=0.35,
                                       default_max=140),
        }
        self._lock = LockDetector()
        self._boss = BossHPDetector()
        self._enemy = EnemyHPDetector()

        self._prev_hp = None
        self._prev_boss = None
        self._last_debug = {}

    def detect(self, frame) -> dict:
        result = {
            "hp": -1.0, "fp": -1.0, "stamina": -1.0,
            "boss_hp": -1.0, "enemy_hp": -1.0,
            "locked": False, "hp_delta": 0.0, "boss_hp_delta": 0.0,
        }
        self._last_debug = {}

        for name in ["hp", "fp", "stamina"]:
            val, dbg = self._detectors[name].detect(frame)
            result[name] = val
            self._last_debug[name] = dbg

        if result["hp"] >= 0 and self._prev_hp is not None and self._prev_hp >= 0:
            result["hp_delta"] = result["hp"] - self._prev_hp
        if result["hp"] >= 0:
            self._prev_hp = result["hp"]

        lock_val, lock_dbg = self._lock.detect(frame)
        result["locked"] = lock_val >= 0.5
        self._last_debug["lock"] = lock_dbg

        boss_val, boss_dbg = self._boss.detect(frame)
        self._last_debug["boss_hp"] = boss_dbg
        if boss_val >= 0:
            result["boss_hp"] = boss_val
            if self._prev_boss is not None and self._prev_boss >= 0:
                result["boss_hp_delta"] = result["boss_hp"] - self._prev_boss
            self._prev_boss = result["boss_hp"]

        if result["locked"]:
            enemy_val, enemy_dbg = self._enemy.detect(frame)
            self._last_debug["enemy_hp"] = enemy_dbg
            if enemy_val >= 0:
                result["enemy_hp"] = enemy_val

        return result

    def get_debug_images(self) -> dict:
        imgs = {}
        for name, det in self._detectors.items():
            dbg = self._last_debug.get(name, {})
            imgs[name] = det.get_debug_image(dbg.get("roi"), dbg)
        return imgs

    @property
    def prev_hp(self):
        return self._prev_hp
