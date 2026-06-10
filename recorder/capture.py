import time
import threading

import dxcam
import numpy as np

from . import config


class Capture:
    """DXCam 截图线程 — 低 GPU 压力。"""

    def __init__(self, target_fps: int = config.FPS):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.camera = dxcam.create(
            output_color=config.DXCAM_OUTPUT_COLOR,
            max_buffer_len=8,  # 小缓冲减少 GPU 内存压力
        )
        self._latest: np.ndarray | None = None
        self._latest_idx: int = -1
        self._latest_ts: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self):
        idx = 0
        while self._running:
            t0 = time.perf_counter()
            img = self.camera.grab()
            if img is not None:
                self._latest = img
                self._latest_idx = idx
                self._latest_ts = time.time()
                idx += 1

            # 精确帧率控制，在该睡的时候把 CPU 时间让给游戏
            elapsed = time.perf_counter() - t0
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)

    @property
    def ready(self) -> bool:
        return self._latest is not None

    def read(self) -> tuple[np.ndarray | None, int, float]:
        return self._latest, self._latest_idx, self._latest_ts
