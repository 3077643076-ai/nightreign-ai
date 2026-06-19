"""虚拟手柄全按键测试：逐个按下所有按钮 + 推所有摇杆轴。
用法：
    python test_gamepad.py
打开 Steam 手柄检测页面，看每个输入有没有反应。
"""

import time
import vgamepad as vg

# 所有要测试的按钮
BUTTONS = [
    ("A", "XUSB_GAMEPAD_A"),
    ("B", "XUSB_GAMEPAD_B"),
    ("X", "XUSB_GAMEPAD_X"),
    ("Y", "XUSB_GAMEPAD_Y"),
    ("LB", "XUSB_GAMEPAD_LEFT_SHOULDER"),
    ("RB", "XUSB_GAMEPAD_RIGHT_SHOULDER"),
    ("BACK", "XUSB_GAMEPAD_BACK"),
    ("START", "XUSB_GAMEPAD_START"),
    ("LS", "XUSB_GAMEPAD_LEFT_THUMB"),
    ("RS", "XUSB_GAMEPAD_RIGHT_THUMB"),
    ("DPAD_UP", "XUSB_GAMEPAD_DPAD_UP"),
    ("DPAD_DOWN", "XUSB_GAMEPAD_DPAD_DOWN"),
    ("DPAD_LEFT", "XUSB_GAMEPAD_DPAD_LEFT"),
    ("DPAD_RIGHT", "XUSB_GAMEPAD_DPAD_RIGHT"),
]

# 摇杆轴测试
AXIS_TESTS = [
    ("左摇杆 ↑ (LY=1)",   lambda g: g.left_joystick_float(0.0, 1.0)),
    ("左摇杆 ↓ (LY=-1)",  lambda g: g.left_joystick_float(0.0, -1.0)),
    ("左摇杆 ← (LX=-1)",  lambda g: g.left_joystick_float(-1.0, 0.0)),
    ("左摇杆 → (LX=1)",   lambda g: g.left_joystick_float(1.0, 0.0)),
    ("右摇杆 ↑ (RY=1)",   lambda g: g.right_joystick_float(0.0, 1.0)),
    ("右摇杆 ↓ (RY=-1)",  lambda g: g.right_joystick_float(0.0, -1.0)),
    ("右摇杆 ← (RX=-1)",  lambda g: g.right_joystick_float(-1.0, 0.0)),
    ("右摇杆 → (RX=1)",   lambda g: g.right_joystick_float(1.0, 0.0)),
    ("LT 扳机",            lambda g: g.left_trigger_float(1.0)),
    ("RT 扳机",            lambda g: g.right_trigger_float(1.0)),
]


def main():
    print("=" * 50)
    print("  虚拟手柄 全按键测试")
    print("  打开 Steam → 设置 → 控制器 → 检测设备")
    print("=" * 50)

    gp = vg.VX360Gamepad()

    print("\n--- 按钮测试 ---")
    for name, attr in BUTTONS:
        print(f"[{name}] 按下...", end=" ", flush=True)
        gp.reset()
        setattr(gp, attr, True)
        gp.update()
        time.sleep(1.0)
        print("松开")
        gp.reset()
        gp.update()
        time.sleep(0.3)

    print("\n--- LT/RT 扳机测试 ---")
    for label in ["LT 扳机", "RT 扳机"]:
        print(f"[{label}] 按下...", end=" ", flush=True)
        gp.reset()
        if label == "LT 扳机":
            gp.left_trigger_float(1.0)
        else:
            gp.right_trigger_float(1.0)
        gp.update()
        time.sleep(1.0)
        print("松开")
        gp.reset()
        gp.update()
        time.sleep(0.3)

    print("\n--- 摇杆轴测试 ---")
    for name, fn in AXIS_TESTS:
        print(f"[{name}] 推...", end=" ", flush=True)
        gp.reset()
        fn(gp)
        gp.update()
        time.sleep(1.0)
        print("归中")
        gp.reset()
        gp.update()
        time.sleep(0.3)

    gp.reset()
    gp.update()
    print("\n测试结束！")


if __name__ == "__main__":
    main()
