"""BC 推理脚本：画面 → 模型 → 虚拟手柄。

用法：
    python -m inference.run

F10 = 切换 AI 控制  |  ESC = 退出
"""

import sys
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from pynput import keyboard

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import vgamepad as vg
    HAS_VGAMEPAD = True
except ImportError:
    HAS_VGAMEPAD = False
    print("[WARN] vgamepad not installed, running in preview mode")

try:
    import dxcam
    HAS_DXCAM = True
except ImportError:
    HAS_DXCAM = False
    print("[ERROR] dxcam required")

from models.bc_model import BCModel
from preprocess.dataset import BUTTON_NAMES, AXIS_NAMES
from preprocess.preprocessed_dataset import IMAGENET_MEAN, IMAGENET_STD

CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "best.pt"
IMG_SIZE = 224
BTN_THRESHOLD = 0.0  # logit > 0 → 按下
AXIS_DEADZONE = 0.15

# 按钮名 → vgamepad 属性名（去掉常见前缀）
_BTN_MAP = {
    "A": "XUSB_GAMEPAD_A", "B": "XUSB_GAMEPAD_B",
    "X": "XUSB_GAMEPAD_X", "Y": "XUSB_GAMEPAD_Y",
    "LB": "XUSB_GAMEPAD_LEFT_SHOULDER", "RB": "XUSB_GAMEPAD_RIGHT_SHOULDER",
    "BACK": "XUSB_GAMEPAD_BACK", "START": "XUSB_GAMEPAD_START",
    "LS": "XUSB_GAMEPAD_LEFT_THUMB", "RS": "XUSB_GAMEPAD_RIGHT_THUMB",
    "DPAD_U": "XUSB_GAMEPAD_DPAD_UP", "DPAD_D": "XUSB_GAMEPAD_DPAD_DOWN",
    "DPAD_L": "XUSB_GAMEPAD_DPAD_LEFT", "DPAD_R": "XUSB_GAMEPAD_DPAD_RIGHT",
}


class BCAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.model = BCModel().to(self.device)
        ckpt = torch.load(CHECKPOINT, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"Model loaded: {CHECKPOINT} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")

        # DXCam
        self.cam = dxcam.create(output_color="BGR")
        print(f"DXCam ready: {self.cam}")

        # vgamepad
        self.gamepad = vg.VX360Gamepad() if HAS_VGAMEPAD else None
        if self.gamepad:
            print("Virtual gamepad ready")

        self.running = False
        self.ai_enabled = False
        self.last_frame_time = 0

    def _preprocess(self, frame):
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        # ImageNet normalize
        frame = (frame - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _apply_gamepad(self, btn_logits, axis_vals):
        if self.gamepad is None:
            return

        self.gamepad.reset()

        # 按钮
        btn_probs = torch.sigmoid(btn_logits[0]).cpu().numpy()
        for i, name in enumerate(BUTTON_NAMES):
            if btn_probs[i] > 0.5:  # sigmoid threshold
                attr = _BTN_MAP.get(name)
                if attr:
                    setattr(self.gamepad, attr, True)
            # LT/RT via trigger
            if name == "LT" and btn_probs[i] > 0.5:
                self.gamepad.left_trigger_float(1.0)
            if name == "RT" and btn_probs[i] > 0.5:
                self.gamepad.right_trigger_float(1.0)

        # 摇杆
        ax = axis_vals[0].cpu().numpy()
        lx, ly = ax[0], ax[1]
        rx, ry = ax[2], ax[3]
        if abs(lx) < AXIS_DEADZONE: lx = 0.0
        if abs(ly) < AXIS_DEADZONE: ly = 0.0
        if abs(rx) < AXIS_DEADZONE: rx = 0.0
        if abs(ry) < AXIS_DEADZONE: ry = 0.0
        self.gamepad.left_joystick_float(lx, ly)
        self.gamepad.right_joystick_float(rx, ry)

        self.gamepad.update()

    def _on_key(self, key):
        try:
            if key == keyboard.Key.f10:
                self.ai_enabled = not self.ai_enabled
                state = "ON" if self.ai_enabled else "OFF"
                print(f"\n[AI {state}]")
                if not self.ai_enabled and self.gamepad:
                    self.gamepad.reset()
                    self.gamepad.update()
            elif key == keyboard.Key.esc:
                self.running = False
                return False
        except AttributeError:
            pass

    def run(self):
        print("\nF8 = toggle AI | ESC = exit\n")
        self.running = True

        listener = keyboard.Listener(on_press=self._on_key)
        listener.start()

        while self.running:
            frame = self.cam.grab()
            if frame is None:
                time.sleep(0.002)
                continue

            if self.ai_enabled:
                tensor = self._preprocess(frame)
                with torch.no_grad():
                    btn_logits, axes = self.model(tensor)
                self._apply_gamepad(btn_logits, axes)

            # 限制帧率 ~15fps
            elapsed = time.perf_counter() - self.last_frame_time
            if elapsed < 0.066:
                time.sleep(0.066 - elapsed)
            self.last_frame_time = time.perf_counter()

        listener.stop()
        if self.gamepad:
            self.gamepad.reset()
            self.gamepad.update()
        print("Done.")


def main():
    if not HAS_DXCAM:
        print("dxcam required: pip install dxcam")
        return
    agent = BCAgent()
    agent.run()


if __name__ == "__main__":
    main()
