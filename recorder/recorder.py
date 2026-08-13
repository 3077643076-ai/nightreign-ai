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
from .boss_config import BossConfig
from .episode import EpisodeTracker
from .episode_splitter import EpisodeSplitter
from .keyboard_mouse_reader import KeyboardMouseReader
from .reward import RewardCalculator
from .retry_controller import RetryController


def _save_frame(path: Path, img: np.ndarray):
    if config.IMAGE_FORMAT == "jpg":
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    else:
        cv2.imwrite(str(path), img)


class Recorder:

    def __init__(self, fps: int = config.FPS,
                 output_dir: Path | None = None,
                 game_state_provider=None,
                 input_provider=None,
                 boss_config: BossConfig | None = None,
                 episode_tracker: EpisodeTracker | None = None,
                 reward_calculator: RewardCalculator | None = None,
                 episode_splitter: EpisodeSplitter | None = None,
                 retry_controller: RetryController | None = None,
                 auto_episode: bool = False):
        """game_state_provider: 可选，返回 dict 的可调用对象（如 MemoryReader().read）。
        用于在录制时同步写入 HP/FP/坐标等 ground truth 标签。"""
        self.fps = fps
        self.capture = Capture(target_fps=fps)
        self.input_reader = InputReader()
        self.input_provider = input_provider or KeyboardMouseReader()
        self._game_state_provider = game_state_provider
        self.boss_config = boss_config or BossConfig()
        self.episode_tracker = episode_tracker
        self.reward_calculator = reward_calculator
        self.episode_splitter = episode_splitter
        self.retry_controller = retry_controller
        self.auto_episode = auto_episode

        self._buffer: deque[tuple] = deque(maxlen=600)
        self._lock = threading.Lock()
        self._running = False
        self._capture_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None

        st = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_stamp = st
        self._episode_index = 1
        if auto_episode:
            self.output_root = output_dir or config.DATA_ROOT
            self.output_dir = self.output_root / self._episode_dir_name()
        else:
            self.output_root = output_dir.parent if output_dir is not None else config.DATA_ROOT
            self.output_dir = output_dir or config.DATA_ROOT / f"session_{st}"
        self.frame_count = 0
        self.start_time: float | None = None
        self._has_game_state = False
        self._finished = False

    def _episode_dir_name(self) -> str:
        return f"episode_{self._session_stamp}_{self._episode_index:04d}"

    def _start_new_episode(self, timestamp: float):
        self.output_dir = self.output_root / self._episode_dir_name()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self.start_time = timestamp
        self._finished = False
        self.episode_tracker = EpisodeTracker(
            episode_id=self.output_dir.name,
            boss_config=self.boss_config,
            fps=self.fps,
        )
        self.episode_tracker.start(timestamp)
        if self.reward_calculator is not None:
            self.reward_calculator.reset()

    def _finalize_current_episode(self):
        total_reward = self.reward_calculator.total_reward if self.reward_calculator else 0.0
        if self.episode_tracker is not None:
            meta = self.episode_tracker.to_meta(total_reward=total_reward)
        else:
            duration = time.time() - self.start_time if self.start_time else 0
            meta = {
                "session_id": self.output_dir.name,
                "fps": self.fps,
                "frame_count": self.frame_count,
                "duration_sec": round(duration, 2),
                "created_at": datetime.now().isoformat(),
            }
        meta["deadzones"] = {"stick": config.STICK_DEADZONE, "trigger": config.TRIGGER_DEADZONE}
        meta["has_game_state"] = self._has_game_state
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def start(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capture.start()
        self.input_reader.start()
        self.input_provider.start()

        for _ in range(100):
            if self.capture.ready:
                break
            time.sleep(0.01)

        self._running = True
        self._start_new_episode(time.time())

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
        self.input_provider.stop()

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

        self._finalize_current_episode()

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

            gamepad_state = self.input_reader.get_state()
            keyboard_mouse_state = self.input_provider.get_state()

            # 读取游戏内存状态（如果可用）
            game_state = None
            if self._game_state_provider is not None:
                try:
                    gs = self._game_state_provider()
                    if gs is not None:
                        game_state = gs
                        self._has_game_state = True
                except Exception:
                    pass

            with self._lock:
                self._buffer.append((img, idx, ts, gamepad_state, keyboard_mouse_state, game_state))
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
        gs_path = self.output_dir / "game_state.jsonl"
        reward_path = self.output_dir / "rewards.jsonl"

        need_resize = config.TARGET_WIDTH and config.TARGET_HEIGHT
        target = (config.TARGET_WIDTH, config.TARGET_HEIGHT) if need_resize else None

        with open(log_path, "a", encoding="utf-8") as log, \
             open(gs_path, "a", encoding="utf-8") as gs_log, \
             open(reward_path, "a", encoding="utf-8") as reward_log:
            for img, idx, ts, gamepad_state, keyboard_mouse_state, game_state in batch:
                frame = img.copy()
                if need_resize and (frame.shape[1], frame.shape[0]) != target:
                    frame = cv2.resize(frame, target)

                _save_frame(frames_dir / f"{idx:06d}.{config.IMAGE_FORMAT}", frame)

                log.write(json.dumps({
                    "frame": idx,
                    "timestamp": ts,
                    "gamepad": gamepad_state,
                    "keyboard": keyboard_mouse_state.get("keys", {}),
                    "mouse": {
                        "buttons": keyboard_mouse_state.get("mouse_buttons", {}),
                        "delta": keyboard_mouse_state.get("mouse_delta", {"dx": 0, "dy": 0, "wheel": 0}),
                    },
                }, ensure_ascii=False) + "\n")

                if self.episode_tracker is not None:
                    self.episode_tracker.observe_frame(idx, ts, game_state)

                reward_result = None
                if self.reward_calculator is not None:
                    reward_result = self.reward_calculator.observe(idx, ts, game_state)
                    reward_log.write(json.dumps({
                        "frame": idx,
                        "timestamp": ts,
                        "reward": reward_result.reward,
                        "events": reward_result.events,
                        "result": reward_result.result,
                        "done": reward_result.done,
                    }, ensure_ascii=False) + "\n")

                if game_state is not None:
                    gs_log.write(json.dumps({
                        "frame": idx,
                        "timestamp": ts,
                        **game_state,
                    }, ensure_ascii=False) + "\n")

                self._maybe_finish_episode(ts, reward_result, game_state)

    def _maybe_finish_episode(self, timestamp: float, reward_result, game_state: dict | None):
        if self._finished or self.episode_splitter is None or self.episode_tracker is None:
            return
        finish, end_reason, result = self.episode_splitter.should_finish(
            self.episode_tracker.start_timestamp,
            timestamp,
            reward_result=reward_result,
            game_state=game_state,
        )
        if not finish:
            return
        total_reward = self.reward_calculator.total_reward if self.reward_calculator else 0.0
        self.episode_tracker.finish(end_reason, result, timestamp, total_reward)
        self._finished = True
        if self.retry_controller is not None:
            self.retry_controller.retry()
        if self.auto_episode:
            self._finalize_current_episode()
            self._episode_index += 1
            self._start_new_episode(timestamp)
