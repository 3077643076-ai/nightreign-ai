"""录制工具入口。F8 开始 / F9 停止，Ctrl+C 退出。"""
import sys
import signal
import threading
from pathlib import Path

from pynput import keyboard

from .recorder import Recorder
from . import config

# 尝试导入内存读取器（仅在游戏运行时可用）
_MEMORY_AVAILABLE = False
try:
    from memory_reader import MemoryReader
    _MEMORY_AVAILABLE = True
except ImportError:
    pass


class RecorderOverlay:
    """录制悬浮窗：红色=录着，灰色=没录。通过标志位避免线程问题。"""

    def __init__(self):
        self._pending = None  # None=不用更新, True=开, False=关
        self._running = False
        try:
            import tkinter as tk
            self._root = tk.Tk()
            self._root.title("Recorder")
            self._root.overrideredirect(True)
            self._root.attributes("-topmost", True)
            self._root.attributes("-alpha", 0.75)
            w, h = 160, 36
            sw = self._root.winfo_screenwidth()
            self._root.geometry(f"{w}x{h}+{sw - w - 20}+{70}")
            self._label = tk.Label(
                self._root, text="REC: OFF", font=("Microsoft YaHei", 12, "bold"),
                fg="white", bg="#333333", padx=12, pady=4,
            )
            self._label.pack(fill="both", expand=True)
            self._root.update()
            self._running = True
        except Exception:
            self._root = None

    def request_recording(self, on: bool):
        """线程安全：设置标志位，主循环轮询更新。"""
        self._pending = on

    def poll(self):
        """主线程调用：处理待更新。"""
        if self._root is None or not self._running or self._pending is None:
            return
        if self._pending:
            self._label.config(text="REC: ON", bg="#cc0000")
        else:
            self._label.config(text="REC: OFF", bg="#333333")
        self._root.update()
        self._pending = None

    def destroy(self):
        self._running = False
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass


def main():
    overlay = RecorderOverlay()

    # 尝试连接游戏内存（训练时用，推理不需要）
    game_state_provider = None
    if _MEMORY_AVAILABLE:
        mr = MemoryReader()
        if mr.open():
            game_state_provider = mr.read
            print("[REC] 内存读取已启用 → HP/FP/坐标/卢恩 将写入 game_state.jsonl")
        else:
            print("[REC] 未检测到游戏进程，仅录制画面+手柄输入")

    recorder = Recorder(fps=config.FPS, game_state_provider=game_state_provider)
    lock = threading.Lock()

    def on_press(key):
        try:
            name = key.name if hasattr(key, 'name') else key.char
        except AttributeError:
            return

        if name == config.HOTKEY_START:
            with lock:
                if not recorder._running:
                    recorder.start()
                    overlay.request_recording(True)

        elif name == config.HOTKEY_STOP:
            with lock:
                if recorder._running:
                    recorder.stop()
                    recorder.flush()
                    overlay.request_recording(False)

    print("=" * 50)
    print("  Game AI Recorder — 录制手柄 + 截图")
    print(f"  FPS: {config.FPS}")
    print(f"  输出: {config.DATA_ROOT}")
    print(f"  F8 = 开始录制  |  F9 = 停止并保存")
    print(f"  Ctrl+C = 退出")
    print("=" * 50)

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

    def signal_handler(sig, frame):
        print("\n[REC] 收到退出信号...")
        if recorder._running:
            recorder.stop()
            recorder.flush()
        overlay.destroy()
        listener.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    import time
    while listener.is_alive():
        overlay.poll()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
