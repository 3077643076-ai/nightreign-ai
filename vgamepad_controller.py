"""虚拟手柄控制器：模型输出 → vgamepad 虚拟 Xbox 360 手柄。

相比键盘映射的改进：
  - 每个按钮独立映射，不再出现 A和B都=Shift 这种冲突
  - 摇杆完整支持（左右摇杆都能用）
  - LT/RT 扳机是模拟量，不是二值
  - 游戏直接用原生手柄输入，体验和真手柄一样

依赖：pip install vgamepad
驱动：需要安装 ViGEmBus (https://github.com/nefarius/ViGEmBus/releases)
"""

import vgamepad as vg
from preprocess.dataset import BUTTON_NAMES, AXIS_NAMES

# 模型输出的按钮索引 → vgamepad 属性名（只写有关系的一对一映射）
_BTN_TO_VGAMEPAD = {
    "A":      "XUSB_GAMEPAD_A",
    "B":      "XUSB_GAMEPAD_B",
    "X":      "XUSB_GAMEPAD_X",
    "Y":      "XUSB_GAMEPAD_Y",
    "LB":     "XUSB_GAMEPAD_LEFT_SHOULDER",
    "RB":     "XUSB_GAMEPAD_RIGHT_SHOULDER",
    "BACK":   "XUSB_GAMEPAD_BACK",
    "START":  "XUSB_GAMEPAD_START",
    "LS":     "XUSB_GAMEPAD_LEFT_THUMB",
    "RS":     "XUSB_GAMEPAD_RIGHT_THUMB",
    "DPAD_U": "XUSB_GAMEPAD_DPAD_UP",
    "DPAD_D": "XUSB_GAMEPAD_DPAD_DOWN",
    "DPAD_L": "XUSB_GAMEPAD_DPAD_LEFT",
    "DPAD_R": "XUSB_GAMEPAD_DPAD_RIGHT",
    "GUIDE":  "XUSB_GAMEPAD_GUIDE",
}


class GamepadController:
    """虚拟 Xbox 360 手柄控制器。

    用法：
        ctrl = GamepadController(threshold=0.25)
        ctrl.apply(btn_probs, axes)  # btn_probs: [17], axes: [6]
    """

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold
        self._gp = vg.VX360Gamepad()
        self._btn_map = {}  # 按钮名 → vgamepad attr 字符串

        # 预计算映射：只存实际存在的属性
        for i, name in enumerate(BUTTON_NAMES):
            attr = _BTN_TO_VGAMEPAD.get(name)
            if attr:
                self._btn_map[i] = attr

        # 按钮状态追踪（用于悬浮窗显示）
        self.btn_state = [0.0] * 17  # 当前按钮状态
        self.axis_state = [0.0] * 6   # 当前摇杆状态

    def apply(self, btn_probs, axes, threshold_override=None):
        """应用模型输出到虚拟手柄。

        Args:
            btn_probs: [17] 按钮概率（已 sigmoid，0~1）
            axes: [6] 摇杆值（LX, LY, RX, RY, DPAD_X, DPAD_Y），范围 [-1, 1]
            threshold_override: 可选，覆盖默认阈值
        """
        thresh = threshold_override if threshold_override is not None else self.threshold
        self._gp.reset()

        # ── 按钮：超过阈值的按下 ──
        for i, attr in self._btn_map.items():
            pressed = btn_probs[i] > thresh
            self.btn_state[i] = 1.0 if pressed else 0.0
            if pressed:
                setattr(self._gp, attr, True)

        # ── LT/RT 扳机：用概率值作为力度（0~1）──
        lt_idx = BUTTON_NAMES.index("LT")
        rt_idx = BUTTON_NAMES.index("RT")
        lt_val = max(0.0, min(1.0, btn_probs[lt_idx]))
        rt_val = max(0.0, min(1.0, btn_probs[rt_idx]))
        if lt_val > 0.05:
            self._gp.left_trigger_float(lt_val)
        if rt_val > 0.05:
            self._gp.right_trigger_float(rt_val)

        # ── 左摇杆（移动）──
        lx = float(axes[0])
        ly = float(axes[1])
        # 去死区
        if abs(lx) < 0.12:
            lx = 0.0
        if abs(ly) < 0.12:
            ly = 0.0
        self._gp.left_joystick_float(lx, ly)
        self.axis_state[0] = lx
        self.axis_state[1] = ly

        # ── 右摇杆（视角）──
        rx = float(axes[2])
        ry = float(axes[3])
        if abs(rx) < 0.12:
            rx = 0.0
        if abs(ry) < 0.12:
            ry = 0.0
        self.axis_state[2] = rx
        self.axis_state[3] = ry
        if rx != 0.0 or ry != 0.0:
            self._gp.right_joystick_float(rx, ry)

        # ── DPAD（通过按钮映射已处理，这里补充轴值记录）──
        self.axis_state[4] = float(axes[4]) if len(axes) > 4 else 0.0
        self.axis_state[5] = float(axes[5]) if len(axes) > 5 else 0.0

        self._gp.update()

    def reset(self):
        """释放所有按钮和摇杆。"""
        self._gp.reset()
        self._gp.update()
        self.btn_state = [0.0] * 17
        self.axis_state = [0.0] * 6
