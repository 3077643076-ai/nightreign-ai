"""录制工具入口。F8 开始 / F9 停止，Ctrl+C 退出。"""
import sys
import signal
import threading
from pathlib import Path

from pynput import keyboard

from .recorder import Recorder
from . import config


def main():
    recorder = Recorder(fps=config.FPS)
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

        elif name == config.HOTKEY_STOP:
            with lock:
                if recorder._running:
                    recorder.stop()
                    recorder.flush()

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
        listener.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    listener.join()


if __name__ == "__main__":
    main()
