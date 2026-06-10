"""游戏 AI 训练素材录制工具 — 快速入口。

用法:
    python record.py

F8 = 开始录制  |  F9 = 停止并保存  |  Ctrl+C = 退出
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recorder.run import main

if __name__ == "__main__":
    main()
