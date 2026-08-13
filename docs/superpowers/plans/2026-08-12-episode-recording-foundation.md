# Episode Recording Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a weekend-testable recording foundation that captures keyboard/mouse input, attaches boss/session metadata, records game state, computes simple episode outcomes, and writes episode-shaped outputs without requiring live RL training.

**Architecture:** Keep input capture, boss metadata, episode writing, outcome detection, and reward scoring as separate small modules under `recorder/`. The existing `Recorder` remains responsible for sampling frames and writing batches; new collaborators provide keyboard/mouse state, boss metadata, episode lifecycle metadata, and reward summaries.

**Tech Stack:** Python 3.10+, `pynput`, `pytest`, existing `dxcam`/OpenCV capture stack, existing optional `MemoryReader` provider.

## Global Constraints

- Do not depend on OCR for Boss identity in the first version; record `boss_id` from manual config.
- Do not replace existing gamepad recording; add keyboard/mouse recording beside it.
- Do not require the game to run for unit tests; use fake frames, fake input states, and fake game states.
- Do not implement DQN/PPO in this plan; only prepare data for later RL training.
- Keep memory reading optional; recording must still work when `MemoryReader.open()` fails.
- Preserve current F8 start / F9 stop workflow.
- Do not make model inference depend on memory values; memory values are training/debug labels.
- Commit steps in this plan are execution checkpoints only; do not run `git commit` unless the user explicitly approves commits.

---

## File Structure

- Create: `recorder/keyboard_mouse_reader.py`
  - Owns real-time keyboard and mouse state snapshots using `pynput` listeners.
  - Produces a normalized dict with `keys`, `mouse_buttons`, and `mouse_delta`.

- Create: `tests/test_keyboard_mouse_reader.py`
  - Unit-tests keyboard/mouse state transitions without starting global OS listeners.

- Create: `recorder/boss_config.py`
  - Owns manually supplied Boss metadata.
  - Loads config from environment variables and optional JSON file.
  - Produces serializable metadata for `meta.json`.

- Create: `tests/test_boss_config.py`
  - Tests defaults, environment overrides, and JSON config loading.

- Create: `recorder/reward.py`
  - Computes per-frame reward and episode summary from memory/game-state samples.
  - First version uses `hp_pct`, `boss_hp_pct`, death, timeout, and elapsed time.

- Create: `tests/test_reward.py`
  - Tests boss damage reward, player damage penalty, kill reward, death penalty, timeout result.

- Create: `recorder/episode.py`
  - Defines `EpisodeMetadata` and `EpisodeTracker`.
  - Tracks start/end timestamps, frame counts, end reason, result, final HP/Boss HP, and reward total.

- Create: `tests/test_episode.py`
  - Tests lifecycle: start, observe frames, finish win/death/timeout, produce meta dict.

- Modify: `recorder/recorder.py`
  - Add optional `input_provider`, `boss_metadata`, `episode_tracker`, and `reward_calculator` parameters.
  - Continue writing existing `inputs.jsonl`, but include `gamepad`, `keyboard`, and `mouse` sections.
  - Write `game_state.jsonl` when a game-state provider exists.
  - Write `rewards.jsonl` when reward calculation is enabled.
  - Extend `meta.json` with boss metadata and episode summary.

- Modify: `recorder/run.py`
  - Wire `KeyboardMouseReader`, `BossConfig`, `EpisodeTracker`, and `RewardCalculator` into `Recorder`.
  - Keep memory reader optional.
  - Keep overlay and F8/F9 controls unchanged.

- Create: `tests/test_recorder_batch.py`
  - Tests batch writing without live screen capture by directly calling `_write_batch()` with fake frame arrays.

---

### Task 1: Keyboard/Mouse State Reader

**Files:**
- Create: `recorder/keyboard_mouse_reader.py`
- Test: `tests/test_keyboard_mouse_reader.py`

**Interfaces:**
- Produces: `KeyboardMouseReader.start() -> KeyboardMouseReader`
- Produces: `KeyboardMouseReader.stop() -> None`
- Produces: `KeyboardMouseReader.get_state() -> dict`
- Produces: `KeyboardMouseReader.record_key_down(name: str) -> None`
- Produces: `KeyboardMouseReader.record_key_up(name: str) -> None`
- Produces: `KeyboardMouseReader.record_mouse_move(dx: int, dy: int) -> None`
- Produces: `KeyboardMouseReader.record_mouse_button(name: str, pressed: bool) -> None`

- [ ] **Step 1: Write failing keyboard/mouse tests**

Create `tests/test_keyboard_mouse_reader.py`:

```python
from recorder.keyboard_mouse_reader import KeyboardMouseReader


def test_key_press_and_release_updates_snapshot():
    reader = KeyboardMouseReader(start_listeners=False)

    reader.record_key_down("w")
    state = reader.get_state()
    assert state["keys"]["w"] == 1

    reader.record_key_up("w")
    state = reader.get_state()
    assert state["keys"]["w"] == 0


def test_mouse_delta_is_accumulated_then_reset_after_snapshot():
    reader = KeyboardMouseReader(start_listeners=False)

    reader.record_mouse_move(5, -3)
    reader.record_mouse_move(2, 4)

    first = reader.get_state()
    second = reader.get_state()

    assert first["mouse_delta"] == {"dx": 7, "dy": 1, "wheel": 0}
    assert second["mouse_delta"] == {"dx": 0, "dy": 0, "wheel": 0}


def test_mouse_button_state_is_recorded():
    reader = KeyboardMouseReader(start_listeners=False)

    reader.record_mouse_button("left", True)
    state = reader.get_state()
    assert state["mouse_buttons"]["left"] == 1

    reader.record_mouse_button("left", False)
    state = reader.get_state()
    assert state["mouse_buttons"]["left"] == 0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_keyboard_mouse_reader.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'recorder.keyboard_mouse_reader'`.

- [ ] **Step 3: Implement minimal keyboard/mouse reader**

Create `recorder/keyboard_mouse_reader.py`:

```python
import threading
from collections import defaultdict

from pynput import keyboard, mouse


class KeyboardMouseReader:
    def __init__(self, start_listeners: bool = True):
        self._start_listeners = start_listeners
        self._keys = defaultdict(int)
        self._mouse_buttons = defaultdict(int)
        self._mouse_delta = {"dx": 0, "dy": 0, "wheel": 0}
        self._last_mouse_pos = None
        self._lock = threading.Lock()
        self._keyboard_listener = None
        self._mouse_listener = None

    def start(self) -> "KeyboardMouseReader":
        if not self._start_listeners:
            return self
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._mouse_listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._keyboard_listener.daemon = True
        self._mouse_listener.daemon = True
        self._keyboard_listener.start()
        self._mouse_listener.start()
        return self

    def stop(self):
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None

    def record_key_down(self, name: str):
        with self._lock:
            self._keys[self._normalize_key_name(name)] = 1

    def record_key_up(self, name: str):
        with self._lock:
            self._keys[self._normalize_key_name(name)] = 0

    def record_mouse_move(self, dx: int, dy: int):
        with self._lock:
            self._mouse_delta["dx"] += int(dx)
            self._mouse_delta["dy"] += int(dy)

    def record_mouse_wheel(self, dy: int):
        with self._lock:
            self._mouse_delta["wheel"] += int(dy)

    def record_mouse_button(self, name: str, pressed: bool):
        with self._lock:
            self._mouse_buttons[self._normalize_mouse_button(name)] = 1 if pressed else 0

    def get_state(self) -> dict:
        with self._lock:
            state = {
                "keys": dict(self._keys),
                "mouse_buttons": dict(self._mouse_buttons),
                "mouse_delta": dict(self._mouse_delta),
            }
            self._mouse_delta = {"dx": 0, "dy": 0, "wheel": 0}
            return state

    def _on_press(self, key):
        self.record_key_down(self._key_to_name(key))

    def _on_release(self, key):
        self.record_key_up(self._key_to_name(key))

    def _on_move(self, x, y):
        with self._lock:
            if self._last_mouse_pos is None:
                self._last_mouse_pos = (x, y)
                return
            last_x, last_y = self._last_mouse_pos
            self._last_mouse_pos = (x, y)
        self.record_mouse_move(x - last_x, y - last_y)

    def _on_click(self, _x, _y, button, pressed):
        self.record_mouse_button(str(button).replace("Button.", ""), pressed)

    def _on_scroll(self, _x, _y, _dx, dy):
        self.record_mouse_wheel(dy)

    def _key_to_name(self, key) -> str:
        if hasattr(key, "char") and key.char:
            return key.char.lower()
        if hasattr(key, "name"):
            return key.name.lower()
        return str(key).replace("Key.", "").lower()

    def _normalize_key_name(self, name: str) -> str:
        return name.replace("Key.", "").lower()

    def _normalize_mouse_button(self, name: str) -> str:
        return name.replace("Button.", "").lower()
```

- [ ] **Step 4: Run keyboard/mouse tests**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_keyboard_mouse_reader.py -v
```

Expected: PASS.

- [ ] **Step 5: Review checkpoint**

Check:

```bash
git -C "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" diff -- recorder/keyboard_mouse_reader.py tests/test_keyboard_mouse_reader.py
```

Expected: only the new reader and tests changed.

---

### Task 2: Boss Metadata Configuration

**Files:**
- Create: `recorder/boss_config.py`
- Test: `tests/test_boss_config.py`

**Interfaces:**
- Produces: `BossConfig` dataclass with fields `boss_id`, `boss_type`, `difficulty`, `weapon`, `required_action`, `control`
- Produces: `BossConfig.to_dict() -> dict`
- Produces: `load_boss_config(path: str | None = None, env: dict | None = None) -> BossConfig`

- [ ] **Step 1: Write failing boss config tests**

Create `tests/test_boss_config.py`:

```python
import json

from recorder.boss_config import BossConfig, load_boss_config


def test_default_boss_config_is_manual_unknown():
    config = load_boss_config(env={})
    assert config.to_dict() == {
        "boss_id": "unknown_boss",
        "boss_type": "standard",
        "difficulty": "normal",
        "weapon": "unknown_weapon",
        "required_action": None,
        "control": "keyboard_mouse",
        "label_source": "manual",
    }


def test_environment_overrides_boss_config():
    config = load_boss_config(env={
        "NIGHTREIGN_BOSS_ID": "tree_boss",
        "NIGHTREIGN_BOSS_TYPE": "mechanic",
        "NIGHTREIGN_DIFFICULTY": "normal",
        "NIGHTREIGN_WEAPON": "storm_weapon",
        "NIGHTREIGN_REQUIRED_ACTION": "weapon_art",
    })
    assert config.boss_id == "tree_boss"
    assert config.boss_type == "mechanic"
    assert config.required_action == "weapon_art"


def test_json_file_overrides_defaults(tmp_path):
    path = tmp_path / "boss.json"
    path.write_text(json.dumps({
        "boss_id": "grafted_scion",
        "boss_type": "standard",
        "difficulty": "normal",
        "weapon": "greatsword",
        "required_action": None,
    }), encoding="utf-8")

    config = load_boss_config(path=str(path), env={})
    assert config.boss_id == "grafted_scion"
    assert config.weapon == "greatsword"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_boss_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'recorder.boss_config'`.

- [ ] **Step 3: Implement boss config**

Create `recorder/boss_config.py`:

```python
from dataclasses import dataclass
import json
import os


@dataclass(frozen=True)
class BossConfig:
    boss_id: str = "unknown_boss"
    boss_type: str = "standard"
    difficulty: str = "normal"
    weapon: str = "unknown_weapon"
    required_action: str | None = None
    control: str = "keyboard_mouse"
    label_source: str = "manual"

    def to_dict(self) -> dict:
        return {
            "boss_id": self.boss_id,
            "boss_type": self.boss_type,
            "difficulty": self.difficulty,
            "weapon": self.weapon,
            "required_action": self.required_action,
            "control": self.control,
            "label_source": self.label_source,
        }


def load_boss_config(path: str | None = None, env: dict | None = None) -> BossConfig:
    values = BossConfig().to_dict()

    if path:
        with open(path, "r", encoding="utf-8") as f:
            file_values = json.load(f)
        for key in values:
            if key in file_values:
                values[key] = file_values[key]

    source_env = os.environ if env is None else env
    env_map = {
        "boss_id": "NIGHTREIGN_BOSS_ID",
        "boss_type": "NIGHTREIGN_BOSS_TYPE",
        "difficulty": "NIGHTREIGN_DIFFICULTY",
        "weapon": "NIGHTREIGN_WEAPON",
        "required_action": "NIGHTREIGN_REQUIRED_ACTION",
        "control": "NIGHTREIGN_CONTROL",
    }
    for key, env_name in env_map.items():
        if source_env.get(env_name):
            values[key] = source_env[env_name]

    return BossConfig(**values)
```

- [ ] **Step 4: Run boss config tests**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_boss_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Review checkpoint**

Run:

```bash
git -C "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" diff -- recorder/boss_config.py tests/test_boss_config.py
```

Expected: only boss config files changed.

---

### Task 3: Reward Calculator

**Files:**
- Create: `recorder/reward.py`
- Test: `tests/test_reward.py`

**Interfaces:**
- Produces: `RewardResult` dataclass with fields `reward: float`, `events: list[str]`, `result: str | None`, `done: bool`
- Produces: `RewardCalculator.observe(frame: int, timestamp: float, game_state: dict | None) -> RewardResult`
- Produces: `RewardCalculator.total_reward -> float`

- [ ] **Step 1: Write failing reward tests**

Create `tests/test_reward.py`:

```python
from recorder.reward import RewardCalculator


def test_boss_hp_drop_gives_positive_reward():
    calc = RewardCalculator(time_penalty=0.0)
    calc.observe(1, 1.0, {"hp_pct": 1.0, "boss_hp_pct": 1.0})
    result = calc.observe(2, 2.0, {"hp_pct": 1.0, "boss_hp_pct": 0.90})
    assert result.reward == 10.0
    assert "boss_damage" in result.events


def test_player_hp_drop_gives_penalty():
    calc = RewardCalculator(time_penalty=0.0)
    calc.observe(1, 1.0, {"hp_pct": 1.0, "boss_hp_pct": 1.0})
    result = calc.observe(2, 2.0, {"hp_pct": 0.80, "boss_hp_pct": 1.0})
    assert result.reward == -20.0
    assert "player_damage" in result.events


def test_boss_dead_finishes_episode_as_win():
    calc = RewardCalculator(time_penalty=0.0)
    calc.observe(1, 1.0, {"hp_pct": 1.0, "boss_hp_pct": 0.10})
    result = calc.observe(2, 2.0, {"hp_pct": 1.0, "boss_hp_pct": 0.0})
    assert result.done is True
    assert result.result == "win"
    assert "boss_dead" in result.events
    assert result.reward == 60.0


def test_player_dead_finishes_episode_as_death():
    calc = RewardCalculator(time_penalty=0.0)
    calc.observe(1, 1.0, {"hp_pct": 0.5, "boss_hp_pct": 0.8})
    result = calc.observe(2, 2.0, {"hp_pct": 0.0, "boss_hp_pct": 0.8})
    assert result.done is True
    assert result.result == "death"
    assert "player_dead" in result.events
    assert result.reward == -150.0


def test_missing_state_only_applies_time_penalty():
    calc = RewardCalculator(time_penalty=-0.01)
    result = calc.observe(1, 1.0, None)
    assert result.reward == -0.01
    assert result.events == ["missing_state"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_reward.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'recorder.reward'`.

- [ ] **Step 3: Implement reward calculator**

Create `recorder/reward.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardResult:
    reward: float
    events: list[str]
    result: str | None
    done: bool


class RewardCalculator:
    def __init__(self, boss_damage_scale: float = 100.0,
                 player_damage_scale: float = -100.0,
                 kill_reward: float = 50.0,
                 death_penalty: float = -100.0,
                 time_penalty: float = -0.01):
        self.boss_damage_scale = boss_damage_scale
        self.player_damage_scale = player_damage_scale
        self.kill_reward = kill_reward
        self.death_penalty = death_penalty
        self.time_penalty = time_penalty
        self._prev_hp_pct = None
        self._prev_boss_hp_pct = None
        self.total_reward = 0.0

    def observe(self, frame: int, timestamp: float, game_state: dict | None) -> RewardResult:
        reward = self.time_penalty
        events = []
        result = None
        done = False

        if game_state is None:
            events.append("missing_state")
            self.total_reward += reward
            return RewardResult(round(reward, 4), events, result, done)

        hp_pct = self._clean_pct(game_state.get("hp_pct"))
        boss_hp_pct = self._clean_pct(game_state.get("boss_hp_pct"))

        if self._prev_boss_hp_pct is not None and boss_hp_pct is not None:
            boss_drop = max(0.0, self._prev_boss_hp_pct - boss_hp_pct)
            if boss_drop > 0:
                reward += boss_drop * self.boss_damage_scale
                events.append("boss_damage")

        if self._prev_hp_pct is not None and hp_pct is not None:
            hp_drop = max(0.0, self._prev_hp_pct - hp_pct)
            if hp_drop > 0:
                reward += hp_drop * self.player_damage_scale
                events.append("player_damage")

        if boss_hp_pct is not None and boss_hp_pct <= 0.0:
            reward += self.kill_reward
            events.append("boss_dead")
            result = "win"
            done = True

        if hp_pct is not None and hp_pct <= 0.0:
            reward += self.death_penalty
            events.append("player_dead")
            result = "death"
            done = True

        if hp_pct is not None:
            self._prev_hp_pct = hp_pct
        if boss_hp_pct is not None:
            self._prev_boss_hp_pct = boss_hp_pct

        self.total_reward += reward
        return RewardResult(round(reward, 4), events, result, done)

    def _clean_pct(self, value):
        if value is None:
            return None
        value = float(value)
        if value < 0:
            return None
        return max(0.0, min(1.0, value))
```

- [ ] **Step 4: Run reward tests**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_reward.py -v
```

Expected: PASS.

- [ ] **Step 5: Review checkpoint**

Run:

```bash
git -C "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" diff -- recorder/reward.py tests/test_reward.py
```

Expected: only reward files changed.

---

### Task 4: Episode Metadata Tracker

**Files:**
- Create: `recorder/episode.py`
- Test: `tests/test_episode.py`

**Interfaces:**
- Consumes: `BossConfig.to_dict() -> dict`
- Consumes: `RewardCalculator.total_reward -> float`
- Produces: `EpisodeTracker.start(timestamp: float) -> None`
- Produces: `EpisodeTracker.observe_frame(frame: int, timestamp: float, game_state: dict | None) -> None`
- Produces: `EpisodeTracker.finish(end_reason: str, result: str, timestamp: float, total_reward: float) -> dict`
- Produces: `EpisodeTracker.to_meta(total_reward: float | None = None) -> dict`

- [ ] **Step 1: Write failing episode tests**

Create `tests/test_episode.py`:

```python
from recorder.boss_config import BossConfig
from recorder.episode import EpisodeTracker


def test_episode_meta_includes_boss_and_final_state():
    tracker = EpisodeTracker(
        episode_id="ep_000001",
        boss_config=BossConfig(boss_id="tree_boss", boss_type="mechanic", weapon="storm_weapon", required_action="weapon_art"),
        fps=15,
    )
    tracker.start(timestamp=10.0)
    tracker.observe_frame(1, 10.1, {"hp_pct": 1.0, "boss_hp_pct": 1.0})
    tracker.observe_frame(2, 12.0, {"hp_pct": 0.8, "boss_hp_pct": 0.0})
    meta = tracker.finish(end_reason="boss_dead", result="win", timestamp=12.0, total_reward=123.5)

    assert meta["episode_id"] == "ep_000001"
    assert meta["boss_id"] == "tree_boss"
    assert meta["boss_type"] == "mechanic"
    assert meta["frame_count"] == 2
    assert meta["duration_sec"] == 2.0
    assert meta["result"] == "win"
    assert meta["end_reason"] == "boss_dead"
    assert meta["final_player_hp"] == 0.8
    assert meta["final_boss_hp"] == 0.0
    assert meta["reward_total"] == 123.5


def test_unfinished_episode_meta_uses_in_progress_status():
    tracker = EpisodeTracker(episode_id="ep_000002", boss_config=BossConfig(), fps=15)
    tracker.start(timestamp=5.0)
    tracker.observe_frame(10, 6.0, None)
    meta = tracker.to_meta(total_reward=-1.0)

    assert meta["result"] == "in_progress"
    assert meta["end_reason"] == "manual_stop"
    assert meta["frame_count"] == 1
    assert meta["reward_total"] == -1.0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_episode.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'recorder.episode'`.

- [ ] **Step 3: Implement episode tracker**

Create `recorder/episode.py`:

```python
from datetime import datetime

from recorder.boss_config import BossConfig


class EpisodeTracker:
    def __init__(self, episode_id: str, boss_config: BossConfig, fps: int):
        self.episode_id = episode_id
        self.boss_config = boss_config
        self.fps = fps
        self.started_at_iso = None
        self.start_timestamp = None
        self.last_timestamp = None
        self.frame_count = 0
        self.final_player_hp = None
        self.final_boss_hp = None
        self.result = "in_progress"
        self.end_reason = "manual_stop"

    def start(self, timestamp: float):
        self.started_at_iso = datetime.now().isoformat()
        self.start_timestamp = timestamp
        self.last_timestamp = timestamp

    def observe_frame(self, frame: int, timestamp: float, game_state: dict | None):
        self.frame_count += 1
        self.last_timestamp = timestamp
        if game_state is None:
            return
        if game_state.get("hp_pct") is not None and game_state.get("hp_pct") >= 0:
            self.final_player_hp = game_state["hp_pct"]
        if game_state.get("boss_hp_pct") is not None and game_state.get("boss_hp_pct") >= 0:
            self.final_boss_hp = game_state["boss_hp_pct"]

    def finish(self, end_reason: str, result: str, timestamp: float, total_reward: float) -> dict:
        self.end_reason = end_reason
        self.result = result
        self.last_timestamp = timestamp
        return self.to_meta(total_reward=total_reward)

    def to_meta(self, total_reward: float | None = None) -> dict:
        duration = 0.0
        if self.start_timestamp is not None and self.last_timestamp is not None:
            duration = max(0.0, self.last_timestamp - self.start_timestamp)

        meta = {
            "episode_id": self.episode_id,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_sec": round(duration, 2),
            "created_at": self.started_at_iso or datetime.now().isoformat(),
            "result": self.result,
            "end_reason": self.end_reason,
            "final_player_hp": self.final_player_hp,
            "final_boss_hp": self.final_boss_hp,
            "reward_total": round(total_reward or 0.0, 4),
        }
        meta.update(self.boss_config.to_dict())
        return meta
```

- [ ] **Step 4: Run episode tests**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_episode.py -v
```

Expected: PASS.

- [ ] **Step 5: Run related tests together**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_boss_config.py tests/test_episode.py -v
```

Expected: PASS.

---

### Task 5: Integrate Keyboard/Mouse, Boss Metadata, Reward, and Episode Meta into Recorder

**Files:**
- Modify: `recorder/recorder.py`
- Test: `tests/test_recorder_batch.py`

**Interfaces:**
- Consumes: `KeyboardMouseReader.get_state() -> dict`
- Consumes: `InputReader.get_state() -> dict`
- Consumes: `EpisodeTracker.observe_frame(frame, timestamp, game_state) -> None`
- Consumes: `EpisodeTracker.to_meta(total_reward) -> dict`
- Consumes: `RewardCalculator.observe(frame, timestamp, game_state) -> RewardResult`
- Produces: `inputs.jsonl` records with `gamepad`, `keyboard`, and `mouse`
- Produces: `rewards.jsonl` records with `frame`, `timestamp`, `reward`, `events`, `result`, `done`
- Produces: extended `meta.json`

- [ ] **Step 1: Write failing recorder batch test**

Create `tests/test_recorder_batch.py`:

```python
import json

import numpy as np

from recorder.boss_config import BossConfig
from recorder.episode import EpisodeTracker
from recorder.recorder import Recorder
from recorder.reward import RewardCalculator


class FakeInputProvider:
    def get_state(self):
        return {
            "keys": {"w": 1},
            "mouse_buttons": {"left": 1},
            "mouse_delta": {"dx": 3, "dy": -1, "wheel": 0},
        }

    def start(self):
        return self

    def stop(self):
        return None


def test_write_batch_includes_gamepad_keyboard_mouse_state_and_reward(tmp_path):
    tracker = EpisodeTracker("ep_test", BossConfig(boss_id="test_boss"), fps=15)
    tracker.start(timestamp=1.0)
    recorder = Recorder(
        fps=15,
        output_dir=tmp_path,
        input_provider=FakeInputProvider(),
        boss_config=BossConfig(boss_id="test_boss"),
        episode_tracker=tracker,
        reward_calculator=RewardCalculator(time_penalty=0.0),
    )

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    gamepad_state = {"buttons": {"A": 1}, "axes": {"LX": 0.5}}
    keyboard_mouse_state = {
        "keys": {"w": 1},
        "mouse_buttons": {"left": 1},
        "mouse_delta": {"dx": 3, "dy": -1, "wheel": 0},
    }
    game_state = {"hp_pct": 1.0, "boss_hp_pct": 1.0}

    recorder._write_batch([(frame, 1, 1.0, gamepad_state, keyboard_mouse_state, game_state)])
    recorder.flush()

    input_line = json.loads((tmp_path / "inputs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert input_line["gamepad"] == gamepad_state
    assert input_line["keyboard"] == {"w": 1}
    assert input_line["mouse"] == {"buttons": {"left": 1}, "delta": {"dx": 3, "dy": -1, "wheel": 0}}

    reward_line = json.loads((tmp_path / "rewards.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert reward_line["frame"] == 1
    assert reward_line["reward"] == 0.0

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["episode_id"] == "ep_test"
    assert meta["boss_id"] == "test_boss"
    assert meta["frame_count"] == 1
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_recorder_batch.py -v
```

Expected: FAIL because `Recorder.__init__()` does not accept `input_provider`, `boss_config`, `episode_tracker`, or `reward_calculator`.

- [ ] **Step 3: Modify Recorder constructor and sampling tuple**

In `recorder/recorder.py`, add imports:

```python
from .boss_config import BossConfig
from .keyboard_mouse_reader import KeyboardMouseReader
from .episode import EpisodeTracker
from .reward import RewardCalculator
```

Change the constructor signature:

```python
def __init__(self, fps: int = config.FPS,
             output_dir: Path | None = None,
             game_state_provider=None,
             input_provider=None,
             boss_config: BossConfig | None = None,
             episode_tracker: EpisodeTracker | None = None,
             reward_calculator: RewardCalculator | None = None):
```

Inside constructor, keep existing gamepad reader and add:

```python
self.input_provider = input_provider or KeyboardMouseReader()
self.boss_config = boss_config or BossConfig()
self.episode_tracker = episode_tracker
self.reward_calculator = reward_calculator
```

In `start()`, after `self.input_reader.start()` add:

```python
self.input_provider.start()
```

After `self.start_time = time.time()` add:

```python
if self.episode_tracker is None:
    self.episode_tracker = EpisodeTracker(
        episode_id=self.output_dir.name,
        boss_config=self.boss_config,
        fps=self.fps,
    )
self.episode_tracker.start(self.start_time)
```

In `stop()`, after `self.input_reader.stop()` add:

```python
self.input_provider.stop()
```

In `_sample_loop()`, change:

```python
state = self.input_reader.get_state()
```

To:

```python
gamepad_state = self.input_reader.get_state()
keyboard_mouse_state = self.input_provider.get_state()
```

Change buffer append to:

```python
self._buffer.append((img, idx, ts, gamepad_state, keyboard_mouse_state, game_state))
```

- [ ] **Step 4: Modify batch writer output shape**

In `_write_batch()`, change loop unpacking:

```python
for img, idx, ts, gamepad_state, keyboard_mouse_state, game_state in batch:
```

Change `inputs.jsonl` write to:

```python
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
```

Before writing game state, add:

```python
if self.episode_tracker is not None:
    self.episode_tracker.observe_frame(idx, ts, game_state)

if self.reward_calculator is not None:
    reward_result = self.reward_calculator.observe(idx, ts, game_state)
    reward_path = self.output_dir / "rewards.jsonl"
    with open(reward_path, "a", encoding="utf-8") as reward_log:
        reward_log.write(json.dumps({
            "frame": idx,
            "timestamp": ts,
            "reward": reward_result.reward,
            "events": reward_result.events,
            "result": reward_result.result,
            "done": reward_result.done,
        }, ensure_ascii=False) + "\n")
```

- [ ] **Step 5: Modify flush meta output**

In `flush()`, replace the current `meta = {...}` block with:

```python
if self.episode_tracker is not None:
    total_reward = self.reward_calculator.total_reward if self.reward_calculator else 0.0
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
```

- [ ] **Step 6: Run recorder batch test**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_recorder_batch.py -v
```

Expected: PASS.

- [ ] **Step 7: Run existing action executor tests to ensure no regression**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_action_executor.py -v
```

Expected: PASS.

---

### Task 6: Wire New Recording Components into CLI Entry

**Files:**
- Modify: `recorder/run.py`
- Test: `tests/test_boss_config.py`, `tests/test_recorder_batch.py`

**Interfaces:**
- Consumes: `load_boss_config(path: str | None = None) -> BossConfig`
- Consumes: `KeyboardMouseReader()`
- Consumes: `RewardCalculator()`
- Consumes: `Recorder(..., input_provider, boss_config, reward_calculator)`

- [ ] **Step 1: Modify imports**

In `recorder/run.py`, add:

```python
from .boss_config import load_boss_config
from .keyboard_mouse_reader import KeyboardMouseReader
from .reward import RewardCalculator
```

- [ ] **Step 2: Load boss config before creating recorder**

In `main()`, before creating `Recorder`, add:

```python
boss_config = load_boss_config()
print(f"[REC] Boss: {boss_config.boss_id} | type={boss_config.boss_type} | difficulty={boss_config.difficulty}")
```

- [ ] **Step 3: Create keyboard/mouse reader and reward calculator**

In `main()`, before creating `Recorder`, add:

```python
keyboard_mouse_reader = KeyboardMouseReader()
reward_calculator = RewardCalculator()
```

- [ ] **Step 4: Pass new components into Recorder**

Replace current recorder construction:

```python
recorder = Recorder(fps=config.FPS, game_state_provider=game_state_provider)
```

With:

```python
recorder = Recorder(
    fps=config.FPS,
    game_state_provider=game_state_provider,
    input_provider=keyboard_mouse_reader,
    boss_config=boss_config,
    reward_calculator=reward_calculator,
)
```

- [ ] **Step 5: Update startup text**

Change:

```python
print("  Game AI Recorder — 录制手柄 + 截图")
```

To:

```python
print("  Game AI Recorder — 录制截图 + 手柄 + 键鼠 + 可选内存状态")
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests/test_boss_config.py tests/test_keyboard_mouse_reader.py tests/test_reward.py tests/test_episode.py tests/test_recorder_batch.py -v
```

Expected: PASS.

---

### Task 7: Add Weekend Smoke Test Instructions to Existing Design Doc

**Files:**
- Modify: `docs/superpowers/specs/2026-08-11-nightreign-episode-rl-design.md`

**Interfaces:**
- Consumes: output files `frames/`, `inputs.jsonl`, `game_state.jsonl`, `rewards.jsonl`, `meta.json`
- Produces: a clear manual test checklist for weekend validation.

- [ ] **Step 1: Append smoke test section**

Append this section to `docs/superpowers/specs/2026-08-11-nightreign-episode-rl-design.md`:

```markdown
## 周末最小验证清单

目标不是训练模型，而是确认录制系统输出正确。

### 启动前

设置 Boss 标签：

```bash
set NIGHTREIGN_BOSS_ID=training_boss
set NIGHTREIGN_BOSS_TYPE=standard
set NIGHTREIGN_DIFFICULTY=normal
set NIGHTREIGN_WEAPON=greatsword
```

### 录制流程

1. 启动游戏和训练场。
2. 运行 `python record.py`。
3. 按 F8 开始录制。
4. 用键鼠打 30-60 秒。
5. 按 F9 停止录制。
6. 打开最新 `data/session_*` 目录。

### 必须检查的输出

- `frames/` 有 jpg 图片。
- `inputs.jsonl` 每行都有 `gamepad`、`keyboard`、`mouse`。
- 如果内存读取成功，`game_state.jsonl` 有 `hp_pct`、`fp_pct`、`stamina_pct`。
- `rewards.jsonl` 存在，并且每行有 `reward`、`events`、`done`。
- `meta.json` 有 `boss_id`、`boss_type`、`difficulty`、`weapon`、`frame_count`、`duration_sec`。

### 成功标准

只要能证明“画面、键鼠、状态、奖励、Boss 标签”在同一帧号附近对齐，就算第一轮成功。
```

- [ ] **Step 2: Review the appended section**

Run:

```bash
git -C "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" diff -- docs/superpowers/specs/2026-08-11-nightreign-episode-rl-design.md
```

Expected: only smoke-test instructions appended.

---

### Task 8: Final Verification

**Files:**
- Verify all changed implementation and test files.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: confidence that weekend recording test can be attempted.

- [ ] **Step 1: Run full available test suite**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && pytest tests -v
```

Expected: PASS.

- [ ] **Step 2: Check syntax for recorder modules**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && python -m compileall recorder control state_machine tests
```

Expected: no syntax errors.

- [ ] **Step 3: Inspect git diff**

Run:

```bash
git -C "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" diff --stat
```

Expected: changes are limited to recorder modules, tests, and the design doc smoke-test section.

- [ ] **Step 4: Manual no-game dry run**

Run:

```bash
cd "C:/Users/fan/Desktop/medev/毕设/nightreign-ai" && python record.py
```

Expected:

```text
[REC] 未检测到游戏进程，仅录制画面+手柄输入
[REC] Boss: unknown_boss | type=standard | difficulty=normal
Game AI Recorder — 录制截图 + 手柄 + 键鼠 + 可选内存状态
```

Press `Ctrl+C` to exit without recording.

---

## Self-Review

**Spec coverage:**
- Keyboard/mouse recording is covered by Task 1 and Task 5.
- Boss manual metadata is covered by Task 2 and Task 6.
- Reward scoring is covered by Task 3 and Task 5.
- Episode-shaped metadata is covered by Task 4 and Task 5.
- Optional memory state remains optional through the existing `game_state_provider` path in Task 5.
- Weekend validation is covered by Task 7 and Task 8.
- Full auto split/retry is intentionally not implemented in this first plan; this plan prepares the data structure and per-session metadata needed before adding automatic episode rollover.

**Placeholder scan:**
- No `TBD`, `TODO`, `implement later`, or vague test instructions remain.

**Type consistency:**
- `BossConfig.to_dict()` is defined in Task 2 and consumed by `EpisodeTracker` in Task 4.
- `RewardCalculator.observe()` returns `RewardResult`, consumed by `Recorder._write_batch()` in Task 5.
- `KeyboardMouseReader.get_state()` shape matches the `inputs.jsonl` output written in Task 5.
