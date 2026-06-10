import json
import time
import threading
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np

from .capture import Capture
from .input_reader import InputReader
from . import config


def _save_frame(path: Path, img: np.ndarray):
    if config.IMAGE_FORMAT == "jpg":
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    else:
        cv2.imwrite(str(path), img)


class Recorder:

    def __init__(self, fps: int = config.FPS,
                 output_dir: Path | None = None):
        self.fps = fps
        self.capture = Capture(target_fps=fps)
        self.input_reader = InputReader()

        self._buffer: deque[tuple] = deque(maxlen=600)
        self._lock = threading.Lock()
        self._running = False
        self._capture_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None

        st = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or config.DATA_ROOT / f"session_{st}"
        self.frame_count = 0
        self.start_time: float | None = None

    def start(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capture.start()
        self.input_reader.start()

        for _ in range(100):
            if self.capture.ready:
                break
            time.sleep(0.01)

        self._running = True
        self.start_time = time.time()

        self._capture_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._capture_thread.start()

        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()

        res = f"{config.TARGET_WIDTH}x{config.TARGET_HEIGHT}" if config.TARGET_WIDTH else "native"
        print(f"[REC] start → {self.output_dir}")
        print(f"       {self.fps}fps | {res} | {config.IMAGE_FORMAT} q{config.JPEG_QUALITY}")

    def stop(self):
        if not self._running:
            return
        self._running = False

        if self._capture_thread:
            self._capture_thread.join(timeout=3.0)
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)

        self.capture.stop()
        self.input_reader.stop()

        duration = time.time() - self.start_time
        avg_fps = self.frame_count / max(duration, 0.01)
        print(f"[REC] stop | {self.frame_count} frames | {duration:.1f}s | avg {avg_fps:.1f} fps")

    def flush(self):
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        with self._lock:
            batch = list(self._buffer)
            self._buffer.clear()

        if batch:
            self._write_batch(batch)

        duration = time.time() - self.start_time if self.start_time else 0
        meta = {
            "session_id": self.output_dir.name,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_sec": round(duration, 2),
            "created_at": datetime.now().isoformat(),
            "deadzones": {"stick": config.STICK_DEADZONE, "trigger": config.TRIGGER_DEADZONE},
        }
        with open(self.output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        ext = config.IMAGE_FORMAT
        print(f"[REC] saved: frames/ ({self.frame_count} {ext}) + inputs.jsonl + meta.json")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()
        self.flush()

    # ── 采样线程（极轻量：只读引用 + 浅拷贝状态，不做任何图像处理）─────

    def _sample_loop(self):
        last_idx = -1
        while self._running:
            img, idx, ts = self.capture.read()
            if img is None or idx == last_idx:
                time.sleep(0.002)
                continue
            last_idx = idx

            state = self.input_reader.get_state()

            with self._lock:
                self._buffer.append((img, idx, ts, state))
                self.frame_count += 1

    # ── 写盘线程（重活全在这：copy + resize + jpeg 编码）─────────

    def _write_loop(self):
        while self._running:
            time.sleep(2.0)
            with self._lock:
                if len(self._buffer) < 30:
                    continue
                batch = list(self._buffer)
                self._buffer.clear()
            self._write_batch(batch)

    def _write_batch(self, batch):
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        log_path = self.output_dir / "inputs.jsonl"

        need_resize = config.TARGET_WIDTH and config.TARGET_HEIGHT
        target = (config.TARGET_WIDTH, config.TARGET_HEIGHT) if need_resize else None

        with open(log_path, "a", encoding="utf-8") as log:
            for img, idx, ts, state in batch:
                # copy + resize（在写盘线程做，不影响游戏）
                frame = img.copy()
                if need_resize and (frame.shape[1], frame.shape[0]) != target:
                    frame = cv2.resize(frame, target)

                _save_frame(frames_dir / f"{idx:06d}.{config.IMAGE_FORMAT}", frame)

                log.write(json.dumps({
                    "frame": idx, "timestamp": ts,
                    "buttons": state["buttons"], "axes": state["axes"],
                }, ensure_ascii=False) + "\n")
