"""键盘控制器：模型输出 → 键盘按键。

映射（黑环默认键位 + 用户实测）：
  X=1(喝血)  A=Shift(闪避)  B=Shift(后撤)
  RB=左键(轻击)  RT=Ctrl+左键(重击,组合)
  LB=右键(防御/左手攻击)
  DPAD←=滚轮下(换左手)  DPAD→=滚轮上(换右手)
  RS=z(锁定!!!)  LS=Alt(疾跑)
  Y+RT=R(绝招)  Y+LT=E(角色技)
  左摇杆=WASD  右摇杆=不映射(鼠标闪)
"""

import numpy as np
from pynput.keyboard import Key, Controller as KBController
from pynput.mouse import Button, Controller as MSController
from preprocess.dataset import BUTTON_NAMES


class KeyboardController:
    def __init__(self, threshold=0.25):
        self.kb = KBController()
        self.ms = MSController()
        self.threshold = threshold
        self._pressed = set()

    def _kpress(self, key):
        if key is not None and key not in self._pressed:
            try:
                if isinstance(key, Button): self.ms.press(key)
                else: self.kb.press(key)
                self._pressed.add(key)
            except: pass

    def _krelease(self, key):
        if key is not None and key in self._pressed:
            try:
                if isinstance(key, Button): self.ms.release(key)
                else: self.kb.release(key)
                self._pressed.discard(key)
            except: pass

    def release_all(self):
        for k in list(self._pressed): self._krelease(k)

    def apply(self, btn_probs, axes):
        """btn_probs: [17], axes: [6] LX,LY,RX,RY,DPAD_X,DPAD_Y"""
        lx, ly = axes[0], axes[1]

        for i, name in enumerate(BUTTON_NAMES):
            p = btn_probs[i] > self.threshold
            if name == "X":       # 喝血 → 1
                self._kpress("1") if p else self._krelease("1")
            elif name == "A":     # 闪避 → Shift
                self._kpress(Key.shift) if p else self._krelease(Key.shift)
            elif name == "B":     # 后撤 → Shift
                self._kpress(Key.shift) if p else self._krelease(Key.shift)
            elif name == "RB":    # 轻击 → 左键
                self._kpress(Button.left) if p else self._krelease(Button.left)
            elif name == "RT":    # 重击 → Ctrl+左键
                if p: self._kpress(Key.ctrl); self._kpress(Button.left)
                else: self._krelease(Key.ctrl); self._krelease(Button.left)
            elif name == "LB":    # 防御 → 右键
                self._kpress(Button.right) if p else self._krelease(Button.right)
            elif name == "LT":    # 防御同
                self._kpress(Button.right) if p else self._krelease(Button.right)
            elif name == "LS":    # 疾跑 → Alt
                self._kpress(Key.alt) if p else self._krelease(Key.alt)
            elif name == "Y":     # 交互 → F（重试！）
                self._kpress("f") if p else self._krelease("f")
            elif name == "RS":    # 锁定 → Z !!
                self._kpress("z") if p else self._krelease("z")
            elif name == "DPAD_L":  # 换左手 → 滚轮下
                if p: self.ms.scroll(0, -1)
            elif name == "DPAD_R":  # 换右手 → 滚轮上
                if p: self.ms.scroll(0, 1)
            # START, BACK, Y, DPAD_U, DPAD_D, GUIDE → 不映射

        # 组合键：Y+RT=R(绝招), Y+LT=E(角色技)
        y = btn_probs[BUTTON_NAMES.index("Y")] > self.threshold
        rt = btn_probs[BUTTON_NAMES.index("RT")] > self.threshold
        lt = btn_probs[BUTTON_NAMES.index("LT")] > self.threshold
        self._kpress("r") if (y and rt) else self._krelease("r")
        self._kpress("e") if (y and lt) else self._krelease("e")

        # 左摇杆 → WASD（不映射右摇杆，避免鼠标闪）
        dead = 0.2
        for key, val in [("w", ly), ("s", -ly), ("a", -lx), ("d", lx)]:
            self._kpress(key) if val > dead else self._krelease(key)

    def reset(self):
        self.release_all()
