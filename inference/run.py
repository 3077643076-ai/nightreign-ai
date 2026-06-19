"""BC 推理脚本：画面 → 模型 → 虚拟手柄。

用法：
    python -m inference.run

F10 = 开启 AI  |  F9 = 关闭 AI  |  F12 = 退出
悬浮窗点击也可切换 AI 状态
"""

import sys
import time
import ctypes
from pathlib import Path

import cv2
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytesseract

# 游戏状态 + 手柄悬浮窗
from controller_overlay import ControllerOverlay
from game_state import GameState

# 优先用虚拟手柄（原生 Xbox 360 映射），键盘作 fallback
HAS_VGAMEPAD = False
try:
    from vgamepad_controller import GamepadController
    HAS_VGAMEPAD = True
except ImportError:
    pass

HAS_KEYBOARD = False
try:
    from keyboard_controller import KeyboardController
    HAS_KEYBOARD = True
except ImportError:
    pass

from pynput.keyboard import Controller as RawKB

try:
    import dxcam
    HAS_DXCAM = True
except ImportError:
    HAS_DXCAM = False

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import tkinter as tk
    HAS_TK = True
except ImportError:
    HAS_TK = False

from models.bc_model import BCModel
from preprocess.dataset import BUTTON_NAMES, AXIS_NAMES
from preprocess.preprocessed_dataset import IMAGENET_MEAN, IMAGENET_STD
from planner import Planner

# ── 分类器模型（和训练时的 Classifier 结构一致）──
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ClassifyModel(nn.Module):
    """轻量二分类器：ResNet-18 → 2 类 (explore/combat)。"""
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(512, 2)

    def forward(self, x):
        # x: [B, 3, H, W] 单帧
        return self.backbone(x)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT = CHECKPOINT_DIR / "best_combat.pt"  # 战斗 v4，含三狼数据
NUM_FRAMES = 4
IMG_SIZE = 224
BTN_THRESHOLD = 0.15
COMBAT_BTN_THRESHOLD = 0.25
# 战斗模式分按钮阈值
DODGE_THRESHOLD = 0.06     # A(闪避)：极低，疯狂躲
ATTACK_THRESHOLD = 0.45     # RB/RT(攻击)：很高，确认能打到才出手
HEAL_THRESHOLD = 0.15       # X(喝药)：中等
AXIS_DEADZONE = 0.15

# ---- 调试开关 ----
SELF_TEST = False        # True=忽略模型，强制推左摇杆上
SAVE_FIRST_FRAME = False  # True=AI 开启时保存第一帧截图
RECORD_SESSION = True     # True=始终录制（手动打也会录，给 AI 做示范）

# ============================================================
# 键盘：用 ctypes GetAsyncKeyState（主线程轮询，无线程问题）
# ============================================================
_GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState
_GetAsyncKeyState.argtypes = [ctypes.c_int]
_GetAsyncKeyState.restype = ctypes.c_short

# 虚拟键码
VK_ESC = 0x1B
VK_F9  = 0x78
VK_F10 = 0x79
VK_F12 = 0x7B  # F12 = 退出（ESC 留给游戏菜单）

# 防抖：记录上一帧按键状态，只在"按下沿"触发
class KeyDebouncer:
    def __init__(self):
        self._prev = {}
    def pressed(self, vk_code):
        """返回 True 仅一次：按键从松开→按下的瞬间。"""
        cur = (_GetAsyncKeyState(vk_code) & 0x8000) != 0
        prev = self._prev.get(vk_code, False)
        self._prev[vk_code] = cur
        return cur and not prev


class _MSSCam:
    """mss 截图包装器。"""
    def __init__(self):
        self._sct = mss.MSS()
        self._monitor = self._sct.monitors[1]
    def grab(self):
        img = np.array(self._sct.grab(self._monitor))
        return img[:, :, :3]


class AIOverlay:
    """悬浮窗：显示 AI 状态，点击切换。"""

    def __init__(self, on_click_toggle):
        self._on_click_toggle = on_click_toggle
        if not HAS_TK:
            self._root = None
            return
        self._root = tk.Tk()
        self._root.title("AI Status")
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", 0.75)
        w, h = 200, 48
        sw = self._root.winfo_screenwidth()
        self._root.geometry(f"{w}x{h}+{sw - w - 20}+{20}")
        self._label = tk.Label(
            self._root, text="AI: OFF", font=("Microsoft YaHei", 14, "bold"),
            fg="white", bg="#333333", padx=16, pady=8, cursor="hand2",
        )
        self._label.pack(fill="both", expand=True)
        self._label.bind("<Button-1>", lambda e: self._on_click_toggle())
        self._root.update()

    def set_state(self, on: bool, loading: bool = False):
        if self._root is None:
            return
        if loading:
            self._label.config(text="AI: LOADING...", bg="#996600")
        elif on:
            self._label.config(text="AI: ON", bg="#1a7a1a")
        else:
            self._label.config(text="AI: OFF", bg="#333333")
        self._root.update()

    def destroy(self):
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass


class BCAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载单模型
        self.model = BCModel(num_frames=NUM_FRAMES).to(self.device)
        ckpt = torch.load(CHECKPOINT, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"Model: {CHECKPOINT.name} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")

        # 规划AI（检测锁定/血条/撞墙）
        self._planner = Planner()

        # 截图、虚拟手柄、悬浮窗、键盘
        self._init_capture_and_gamepad()

    @staticmethod
    def _load_classifier():
        ckpt = torch.load(CHECKPOINT_CLASSIFIER, map_location="cpu", weights_only=False)
        model = ClassifyModel()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"Classifier: {CHECKPOINT_CLASSIFIER.name} "
              f"(epoch {ckpt['epoch']}, acc={ckpt.get('val_acc', '?')})")
        return model

    def _detect_lock_target(self, frame) -> bool:
        """检测画面中心是否有可锁定目标（白色标记点）。
        返回 True 表示可以尝试锁定。"""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        rw, rh = w // 3, h // 3
        x1, y1 = cx - rw // 2, cy - rh // 2
        x2, y2 = cx + rw // 2, cy + rh // 2
        roi = frame[y1:y2, x1:x2]

        white = cv2.inRange(roi, np.array([180, 180, 180]), np.array([255, 255, 255]))
        kernel = np.ones((3, 3), np.uint8)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 至少一个 >60 像素的白块 → 有可锁定目标
        for cnt in contours:
            if cv2.contourArea(cnt) > 60:
                return True
        return False

    def _preprocess_single(self, frame):
        """单帧预处理（给分类器用，无堆叠）。"""
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        # [H, W, 3] → [1, 3, H, W]
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @staticmethod
    def _detect_death_screen(frame) -> bool:
        """OCR 检测死亡画面：识别到'圆桌'或'重试'返回 True。"""
        try:
            h, w = frame.shape[:2]
            # 文字在画面中下方
            roi = frame[h*2//5:h*3//4, w//4:3*w//4]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 二值化：文字是白色的
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            text = pytesseract.image_to_string(thresh, lang='chi_sim')
            keywords = ['圆桌', '重试', '返回']
            for kw in keywords:
                if kw in text:
                    return True
        except Exception:
            pass
        return False

    def _init_capture_and_gamepad(self):
        """初始化截图和虚拟手柄（从 __init__ 分离出来）。"""
        self.cam = None
        if HAS_DXCAM:
            try:
                self.cam = dxcam.create(output_color="BGR")
                print(f"DXCam ready: {self.cam}")
            except Exception as e:
                print(f"[WARN] DXCam 初始化失败: {e}")
        if self.cam is None and HAS_MSS:
            self.cam = _MSSCam()
            print(f"MSS fallback ready: {self.cam}")

        # 虚拟手柄优先 → 键盘 fallback
        self._use_gamepad = False
        if HAS_VGAMEPAD:
            try:
                self._ctrl = GamepadController(threshold=BTN_THRESHOLD)
                # 冒烟测试：reset 一把确认驱动正常
                self._ctrl.reset()
                self._use_gamepad = True
                print("虚拟手柄控制器就绪 (Xbox 360)")
            except Exception as e:
                print(f"[WARN] 虚拟手柄初始化失败: {e}")
                print("  → ViGEmBus 驱动可能没装：https://github.com/nefarius/ViGEmBus/releases")
        if not self._use_gamepad:
            if HAS_KEYBOARD:
                self._ctrl = KeyboardController(threshold=BTN_THRESHOLD)
                print("键盘控制器就绪 (fallback)")
            else:
                raise ImportError("需要 vgamepad 或 keyboard_controller！")
        self._raw_kb = RawKB()  # 用于 F 键等辅助操作

        self._overlay = AIOverlay(on_click_toggle=self._toggle_ai)
        self._monitor = ControllerOverlay()  # 游戏状态 + 手柄监视悬浮窗
        self._game_state = GameState()       # 血条检测器
        self.ai_enabled = False  # 默认关闭，F10 开启
        self._overlay.set_state(False)
        self._last_frame_time = 0
        self._frame_buf = []
        self._debug_cnt = 0
        self._first_frame_saved = False
        self._key_state = KeyDebouncer()

    def _toggle_ai(self):
        """悬浮窗点击：切换 AI 开关。"""
        if self.ai_enabled:
            self._stop_ai()
        else:
            self._start_ai()

    def _start_ai(self):
        """开启 AI，先加载 OWLv2 再开始推理。"""
        if not self.ai_enabled:
            self.ai_enabled = True
            self._overlay.set_state(True, loading=True)
            print("\n[AI LOADING] 正在加载视觉模型...")
            # self._planner._load_owl()  # 砍掉 OWLv2，不加载
            self._debug_cnt = 0
            self._first_frame_saved = False
            self._frame_buf.clear()
            self._overlay.set_state(True)
            # 手柄悬浮窗
            if HAS_OVERLAY and self._ctrl_overlay is None:
                self._ctrl_overlay = ControllerOverlay()
            # 初始化录制
            if RECORD_SESSION:
                self._init_recording()
            print("[AI ON] 就绪！")

    def _init_recording(self):
        """创建录制目录。"""
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._record_dir = Path(__file__).resolve().parent.parent / "preprocessed" / f"session_{ts}"
        (self._record_dir / "frames").mkdir(parents=True, exist_ok=True)
        self._record_fids = []
        self._record_btns = []
        self._record_axes = []
        self._record_combat = []
        self._record_cnt = 0
        print(f"[RECORD] 录制 → {self._record_dir.name}")

    def _save_record_frame(self, frame, btn_logits, axes, combat_label):
        """保存一帧训练数据。"""
        if not hasattr(self, '_record_dir'):
            return
        self._record_cnt += 1
        # 每 3 帧保存一次（~5fps 录制，节省空间）
        if self._record_cnt % 3 != 0:
            return
        fid = self._record_cnt
        cv2.imwrite(str(self._record_dir / "frames" / f"{fid:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        self._record_fids.append(fid)
        btn_p = torch.sigmoid(btn_logits[0]).cpu().numpy()
        self._record_btns.append(btn_p.tolist())
        ax_v = axes[0].cpu().numpy()
        self._record_axes.append(ax_v.tolist())
        self._record_combat.append(int(combat_label))

    def _finish_recording(self):
        """保存录制数据到 JSON + npy。"""
        if not hasattr(self, '_record_dir') or self._record_dir is None:
            return
        import json
        data = {"frame_ids": self._record_fids, "buttons": self._record_btns, "axes": self._record_axes}
        with open(self._record_dir / "labels.json", "w") as f:
            json.dump(data, f)
        np.save(self._record_dir / "combat_labels.npy",
                np.array(self._record_combat, dtype=np.uint8))
        print(f"[RECORD] 已保存 {len(self._record_fids)} 帧 → {self._record_dir.name}")
        self._record_dir = None

    def _stop_ai(self):
        """关闭 AI。"""
        if self.ai_enabled:
            self.ai_enabled = False
            self._frame_buf.clear()
            self._ctrl.reset()
            self._overlay.set_state(False)
            if RECORD_SESSION:
                self._finish_recording()
            if self._ctrl_overlay:
                self._ctrl_overlay.destroy()
                self._ctrl_overlay = None
            print("\n[AI OFF]")

    def _preprocess(self, frame):
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)

        self._frame_buf.append(frame)
        if len(self._frame_buf) > NUM_FRAMES:
            self._frame_buf.pop(0)
        while len(self._frame_buf) < NUM_FRAMES:
            self._frame_buf.insert(0, self._frame_buf[0])

        stacked = np.concatenate(self._frame_buf, axis=2)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _apply_gamepad(self, btn_logits, axis_vals, threshold=None, combat_mode=False):
        """模型输出 → 键盘/鼠标。"""
        ax = axis_vals[0].cpu().numpy()
        lx, ly = ax[0], ax[1]
        rx, ry = ax[2], ax[3]
        if abs(lx) < AXIS_DEADZONE: lx = 0.0
        if abs(ly) < AXIS_DEADZONE: ly = 0.0
        if abs(rx) < AXIS_DEADZONE: rx = 0.0
        if abs(ry) < AXIS_DEADZONE: ry = 0.0
        ax = np.array([lx, ly, rx, ry, ax[4], ax[5]])
        btn_probs = torch.sigmoid(btn_logits[0]).cpu().numpy()
        if combat_mode:
            for i, name in enumerate(BUTTON_NAMES):
                if name == "A": btn_probs[i] = 1.0 if btn_probs[i] > DODGE_THRESHOLD else 0.0
                elif name in ("RB", "RT"): btn_probs[i] = 1.0 if btn_probs[i] > ATTACK_THRESHOLD else 0.0
                elif name == "X": btn_probs[i] = 1.0 if btn_probs[i] > HEAL_THRESHOLD else 0.0
                elif btn_probs[i] < COMBAT_BTN_THRESHOLD: btn_probs[i] = 0.0
        self._ctrl.apply(btn_probs, ax)

    def _process_keys(self):
        """主线程轮询按键（无回调、无线程、无竞争）。"""
        if self._key_state.pressed(VK_F10):
            self._start_ai()
        if self._key_state.pressed(VK_F9):
            self._stop_ai()
        if self._key_state.pressed(VK_F12):
            self.running = False

    def run(self):
        print("\n  F10 = 开启 AI  |  F9 = 关闭 AI  |  F12 = 退出")
        print("  点击右上角悬浮窗也可切换\n")
        self.running = True

        while self.running:
            self._process_keys()

            frame = self.cam.grab()
            if frame is None:
                time.sleep(0.002)
                continue

            # 游戏状态检测 + 悬浮窗更新（不管 AI 开没开都跑）
            gs = self._game_state.detect(frame)
            ctrl_btns = self._ctrl.btn_state if hasattr(self._ctrl, 'btn_state') else None
            ctrl_axes = self._ctrl.axis_state if hasattr(self._ctrl, 'axis_state') else None
            self._monitor.update(btn_state=ctrl_btns, axis_state=ctrl_axes,
                                ai_enabled=self.ai_enabled, game_state=gs)

            if self.ai_enabled:
                tensor = self._preprocess(frame)

                if SELF_TEST:
                    self._ctrl.apply(np.zeros(17), np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
                    self._debug_cnt += 1
                    if self._debug_cnt == 1:
                        print("[SELF-TEST] 强制推左摇杆 ↑ —— 角色应该往前走")
                    if self._debug_cnt % 30 == 0:
                        print(f"[SELF-TEST #{self._debug_cnt}] 仍在发送左摇杆 ↑ ...")
                    if SAVE_FIRST_FRAME and not self._first_frame_saved:
                        cv2.imwrite("debug_frame.png", frame)
                        self._first_frame_saved = True
                        print("[SAVED] 第一帧截图 → debug_frame.png")
                else:
                    # ── 锁圈 / 撞墙 检测 ──
                    if self._planner._h == 0:
                        self._planner._h, self._planner._w = frame.shape[:2]
                    locked = self._planner._detect_lock_circle(frame)
                    stuck = self._planner._detect_stuck(frame)
                    if not hasattr(self._planner, '_lock_cnt'):
                        self._planner._lock_cnt = 0
                    # 锁圈确认需要3帧（防假阳性），丢失需要连续5帧无锁圈
                    self._planner._lock_cnt = self._planner._lock_cnt + 1 if locked else max(0, self._planner._lock_cnt - 1)

                    if stuck:
                        pl_mode = "stuck"
                    elif self._planner._lock_cnt >= 3:
                        pl_mode = "combat"
                    elif self._planner._lock_cnt > 0:
                        pl_mode = "approach"
                    else:
                        pl_mode = "explore"

                    with torch.no_grad():
                        btn_logits, axes = self.model(tensor)

                    self._debug_cnt += 1

                    # ── 锁定辅助：根据模式调整 RS 按钮强度 ──
                    # 直接修改 btn_logits（logit=5.0  → sigmoid≈0.993 > 所有阈值）
                    t_now = time.perf_counter()
                    if not hasattr(self, '_t_rs'): self._t_rs = t_now; self._t_f = t_now
                    lock_confirmed = self._planner._lock_cnt >= 3
                    rs_idx = BUTTON_NAMES.index("RS")

                    if pl_mode == "approach":
                        # 接近模式：持续按 RS 直到锁上（每 0.3s 脉冲一次）
                        rs_on = (t_now - self._t_rs > 0.3)
                    elif pl_mode == "explore":
                        # 探索模式：每 0.8s 试探性按一次 RS
                        rs_on = (t_now - self._t_rs > 0.8)
                    elif pl_mode == "combat" and not lock_confirmed:
                        # 战斗中锁定丢了：立即补锁（0.2s 间隔，最高优先级）
                        rs_on = (t_now - self._t_rs > 0.2)
                    elif pl_mode == "combat":
                        rs_on = False  # 锁定正常：信任模型
                    else:
                        rs_on = False

                    if rs_on:
                        self._t_rs = t_now
                        # 修改原始 logit，确保后续 _apply_gamepad 也能看到
                        btn_logits[0, rs_idx] = 5.0

                    # 用于日志显示的 btn_p（修改后）
                    btn_p = torch.sigmoid(btn_logits[0]).cpu().numpy()
                    ax_v = axes[0].cpu().numpy()

                    # ── F 键（交互/复活）：所有模式生效 ──
                    f_on = (t_now - self._t_f > 1.5)  # 1.5s 间隔，避免卡键
                    if f_on:
                        self._t_f = t_now
                        self._raw_kb.press("f"); self._raw_kb.release("f")

                    # ── 根据模式应用动作 ──
                    if pl_mode == "stuck":
                        # 撞墙：右摇杆强制右转脱困，保留模型其他输出
                        ax_v[2] = 1.0  # 右摇杆右
                        ax_v[3] = 0.0
                        ax_v[1] = -0.3  # 稍微后退
                        self._ctrl.apply(btn_p, ax_v)
                    elif pl_mode == "combat":
                        # 战斗：全模型驱动 + 分阈值
                        self._apply_gamepad(btn_logits, axes, combat_mode=True)
                    else:
                        # 探索/接近：模型驱动，摇杆全开
                        self._apply_gamepad(btn_logits, axes, combat_mode=False)

                    # ── 手柄悬浮窗更新 ──
                    if self._ctrl_overlay and hasattr(self._ctrl, 'btn_state'):
                        self._ctrl_overlay.update(self._ctrl.btn_state, self._ctrl.axis_state)

                    # ── 录制训练数据 ──
                    if RECORD_SESSION:
                        is_combat = pl_mode in ("combat", "approach")
                        self._save_record_frame(frame, btn_logits, axes, is_combat)

                    # ── 调试日志 ──
                    if self._debug_cnt <= 3 or self._debug_cnt % 30 == 0:
                        ax_v_log = axes[0].cpu().numpy()
                        top_idx = np.argsort(btn_p)[::-1][:5]
                        top_info = ", ".join(f"{BUTTON_NAMES[i]}={btn_p[i]:.3f}" for i in top_idx)
                        lc = self._planner._lock_cnt if hasattr(self._planner, '_lock_cnt') else 0
                        lock_str = f" lock={lc}/3" if lc > 0 else ""
                        tag = "FIGHT" if (pl_mode == "combat" and lock_confirmed) else pl_mode[:4].upper()
                        print(f"[{tag} #{self._debug_cnt}{lock_str}] BTN: {top_info}")
                        print(f"[{tag} #{self._debug_cnt}] AXIS: LX={ax_v_log[0]:.3f} "
                              f"LY={ax_v_log[1]:.3f} RX={ax_v_log[2]:.3f} RY={ax_v_log[3]:.3f}")
                        if SAVE_FIRST_FRAME and not self._first_frame_saved:
                            cv2.imwrite("debug_frame.png", frame)
                            self._first_frame_saved = True
                            print("[SAVED] 第一帧截图 → debug_frame.png")

            elapsed = time.perf_counter() - self._last_frame_time
            if elapsed < 0.066:
                time.sleep(0.066 - elapsed)
            self._last_frame_time = time.perf_counter()

        self._ctrl.reset()
        self._monitor.destroy()
        self._overlay.destroy()
        print("Done.")


def main():
    if not HAS_DXCAM and not HAS_MSS:
        print("需要 dxcam 或 mss: pip install dxcam mss")
        return
    agent = BCAgent()
    agent.run()


if __name__ == "__main__":
    main()
