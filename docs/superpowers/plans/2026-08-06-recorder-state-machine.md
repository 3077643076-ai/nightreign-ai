# Recorder And State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first usable data-collection loop: record frames, inputs, memory state, visual state, and state-machine labels while also providing a supervisor that can execute retry/lock-on actions.

**Architecture:** Keep the recorder as the data source and add small focused modules around it. The recorder writes normalized `memory_state.jsonl` and `metadata.json`; the state machine converts visual/memory signals into Chinese-labeled flow states; the supervisor uses those states to execute confirm/lock actions and log Chinese decisions.

**Tech Stack:** Python 3.10+, DXCam, OpenCV, pynput, existing `MemoryReader`, existing `GameState`, jsonl logs.

## Global Constraints

- Only for PVE / offline / practice Mod / personal research environments.
- Company computer must not run the game, CE, or Mod tests; local tests must use fake frames/states only.
- Do not implement RL training in this plan.
- Do not implement route planning, map planning, inventory OCR, or equipment planning in this plan.
- Logs must be human-readable Chinese; JSON keys stay English and Chinese text uses `_cn` fields.
- Real input execution is allowed by design, but every executable action must support `dry_run=True` for safe local testing.
- Existing recorder hotkeys remain F8 start and F9 stop.
- Existing frame/input recording must keep working.

---

## File Structure

- Modify: `recorder/recorder.py` — normalize recorded output files and add visual/memory/state-machine event fields.
- Modify: `recorder/run.py` — wire visual state detector, memory state provider, and metadata flags into `Recorder`.
- Modify: `recorder/config.py` — add state-machine/supervisor key bindings and dry-run defaults.
- Create: `state_machine/__init__.py` — package exports.
- Create: `state_machine/states.py` — state enum and Chinese labels.
- Create: `state_machine/supervisor.py` — deterministic flow-state classifier from visual/memory observations.
- Create: `control/__init__.py` — package exports.
- Create: `control/action_executor.py` — press confirm, lock-on, release-all with `dry_run` support.
- Create: `run_supervisor.py` — live supervisor loop for retry/lock-on execution.
- Create: `tests/test_state_machine.py` — unit tests for state classification.
- Create: `tests/test_action_executor.py` — unit tests for dry-run action logging.
- Create: `tests/test_recorder_schema.py` — unit test for JSON schema normalization helper.

---

### Task 1: State Enum And Classifier

**Files:**
- Create: `state_machine/__init__.py`
- Create: `state_machine/states.py`
- Create: `state_machine/supervisor.py`
- Test: `tests/test_state_machine.py`

**Interfaces:**
- Consumes: visual state dict from `GameState.detect(frame)` when available; memory dict from `MemoryReader.read()` when available.
- Produces: `FlowDecision` dataclass with `state: FlowState`, `state_cn: str`, `confidence: float`, `reason_cn: str`, `should_release_inputs: bool`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_state_machine.py`:

```python
from state_machine.supervisor import StateSupervisor
from state_machine.states import FlowState


def test_dead_state_from_memory_hp_zero():
    supervisor = StateSupervisor()
    decision = supervisor.classify(
        visual_state={},
        memory_state={"hp": 0, "hp_pct": 0.0},
    )
    assert decision.state == FlowState.DEAD
    assert decision.state_cn == "死亡"
    assert "血量" in decision.reason_cn
    assert decision.should_release_inputs is True


def test_combat_state_from_boss_hp_or_lock():
    supervisor = StateSupervisor()
    decision = supervisor.classify(
        visual_state={"boss_hp": 0.75, "locked": True},
        memory_state={"hp": 800, "hp_pct": 0.8},
    )
    assert decision.state == FlowState.COMBAT
    assert decision.state_cn == "战斗中"
    assert decision.confidence >= 0.7
    assert decision.should_release_inputs is False


def test_unknown_state_when_no_signal():
    supervisor = StateSupervisor()
    decision = supervisor.classify(visual_state={}, memory_state=None)
    assert decision.state == FlowState.UNKNOWN
    assert decision.state_cn == "未知"
    assert decision.should_release_inputs is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state_machine.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'state_machine'`.

- [ ] **Step 3: Implement states**

Create `state_machine/states.py`:

```python
from enum import Enum


class FlowState(str, Enum):
    LOADING = "loading"
    MENU = "menu"
    DEAD = "dead"
    RETRY_PROMPT = "retry_prompt"
    COMBAT = "combat"
    REWARD = "reward"
    UNKNOWN = "unknown"


STATE_CN = {
    FlowState.LOADING: "加载中",
    FlowState.MENU: "菜单中",
    FlowState.DEAD: "死亡",
    FlowState.RETRY_PROMPT: "再战确认",
    FlowState.COMBAT: "战斗中",
    FlowState.REWARD: "结算奖励",
    FlowState.UNKNOWN: "未知",
}
```

- [ ] **Step 4: Implement classifier**

Create `state_machine/supervisor.py`:

```python
from dataclasses import dataclass
from typing import Any

from .states import FlowState, STATE_CN


@dataclass(frozen=True)
class FlowDecision:
    state: FlowState
    state_cn: str
    confidence: float
    reason_cn: str
    should_release_inputs: bool


class StateSupervisor:
    def classify(self, visual_state: dict[str, Any] | None, memory_state: dict[str, Any] | None) -> FlowDecision:
        visual_state = visual_state or {}
        memory_state = memory_state or {}

        hp = memory_state.get("hp")
        hp_pct = memory_state.get("hp_pct")
        if hp == 0 or (isinstance(hp_pct, (int, float)) and hp_pct <= 0.01):
            return self._decision(
                FlowState.DEAD,
                0.95,
                "内存状态显示玩家血量为0，判定为死亡，需要停止战斗输入",
                True,
            )

        if visual_state.get("retry_available") or memory_state.get("retry_available"):
            return self._decision(
                FlowState.RETRY_PROMPT,
                0.9,
                "检测到再战确认提示，可以执行确认输入",
                True,
            )

        boss_hp = visual_state.get("boss_hp", visual_state.get("boss_hp_pct"))
        locked = visual_state.get("locked", visual_state.get("locked_on", memory_state.get("locked_on")))
        if self._valid_ratio(boss_hp) or locked is True:
            return self._decision(
                FlowState.COMBAT,
                0.8,
                "检测到Boss血条或锁定标记，判定为战斗中",
                False,
            )

        if visual_state.get("in_menu") or memory_state.get("in_menu"):
            return self._decision(
                FlowState.MENU,
                0.75,
                "检测到菜单状态，暂停战斗输入",
                True,
            )

        return self._decision(
            FlowState.UNKNOWN,
            0.2,
            "没有足够的视觉或内存信号，保持安全等待",
            True,
        )

    def _decision(self, state: FlowState, confidence: float, reason_cn: str, should_release_inputs: bool) -> FlowDecision:
        return FlowDecision(
            state=state,
            state_cn=STATE_CN[state],
            confidence=confidence,
            reason_cn=reason_cn,
            should_release_inputs=should_release_inputs,
        )

    def _valid_ratio(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and 0.0 < float(value) <= 1.0
```

Create `state_machine/__init__.py`:

```python
from .states import FlowState
from .supervisor import FlowDecision, StateSupervisor

__all__ = ["FlowState", "FlowDecision", "StateSupervisor"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_state_machine.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add state_machine tests/test_state_machine.py
git commit -m "feat: add flow state classifier"
```

---

### Task 2: Dry-Run Safe Action Executor

**Files:**
- Create: `control/__init__.py`
- Create: `control/action_executor.py`
- Modify: `recorder/config.py`
- Test: `tests/test_action_executor.py`

**Interfaces:**
- Consumes: key bindings from `recorder.config`.
- Produces: `ActionExecutor.press_confirm()`, `ActionExecutor.lock_on()`, `ActionExecutor.release_all()`, each returns `ActionResult` with Chinese summary.

- [ ] **Step 1: Write the failing test**

Create `tests/test_action_executor.py`:

```python
from control.action_executor import ActionExecutor


def test_dry_run_confirm_returns_chinese_summary():
    executor = ActionExecutor(dry_run=True, confirm_key="f", lock_on_key="q")
    result = executor.press_confirm()
    assert result.action == "confirm"
    assert result.executed is False
    assert "确认" in result.summary_cn
    assert "dry-run" in result.summary_cn


def test_dry_run_lock_on_returns_chinese_summary():
    executor = ActionExecutor(dry_run=True, confirm_key="f", lock_on_key="q")
    result = executor.lock_on()
    assert result.action == "lock_on"
    assert result.executed is False
    assert "锁定" in result.summary_cn


def test_release_all_is_safe_in_dry_run():
    executor = ActionExecutor(dry_run=True, confirm_key="f", lock_on_key="q")
    result = executor.release_all()
    assert result.action == "release_all"
    assert result.executed is False
    assert "释放" in result.summary_cn
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_action_executor.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'control'`.

- [ ] **Step 3: Add config values**

Append to `recorder/config.py`:

```python
# 状态机 / 规划脑按键
SUPERVISOR_DRY_RUN = False
CONFIRM_KEY = "f"              # 死亡后再战 / 菜单确认
LOCK_ON_KEY = "q"              # 锁定目标，按个人键位修改
SUPERVISOR_INTERVAL_SEC = 0.5
RETRY_CONFIRM_DELAY_SEC = 0.8
```

- [ ] **Step 4: Implement executor**

Create `control/action_executor.py`:

```python
from dataclasses import dataclass
import time

from pynput.keyboard import Controller, Key


@dataclass(frozen=True)
class ActionResult:
    action: str
    executed: bool
    summary_cn: str


class ActionExecutor:
    def __init__(self, dry_run: bool, confirm_key: str, lock_on_key: str, press_duration: float = 0.05):
        self.dry_run = dry_run
        self.confirm_key = confirm_key
        self.lock_on_key = lock_on_key
        self.press_duration = press_duration
        self._keyboard = Controller()

    def press_confirm(self) -> ActionResult:
        return self._press("confirm", self.confirm_key, "确认/再战")

    def lock_on(self) -> ActionResult:
        return self._press("lock_on", self.lock_on_key, "锁定目标")

    def release_all(self) -> ActionResult:
        if self.dry_run:
            return ActionResult("release_all", False, "[动作执行] dry-run：释放所有按键，防止卡键")
        for key in (self.confirm_key, self.lock_on_key):
            self._keyboard.release(self._to_key(key))
        return ActionResult("release_all", True, "[动作执行] 已释放确认键和锁定键，防止卡键")

    def _press(self, action: str, key: str, label_cn: str) -> ActionResult:
        if self.dry_run:
            return ActionResult(action, False, f"[动作执行] dry-run：按下 {key}（{label_cn}）")
        normalized = self._to_key(key)
        self._keyboard.press(normalized)
        time.sleep(self.press_duration)
        self._keyboard.release(normalized)
        return ActionResult(action, True, f"[动作执行] 已按下 {key}（{label_cn}）")

    def _to_key(self, key: str):
        special = {
            "space": Key.space,
            "enter": Key.enter,
            "esc": Key.esc,
            "tab": Key.tab,
        }
        return special.get(key.lower(), key)
```

Create `control/__init__.py`:

```python
from .action_executor import ActionExecutor, ActionResult

__all__ = ["ActionExecutor", "ActionResult"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_action_executor.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add control recorder/config.py tests/test_action_executor.py
git commit -m "feat: add supervisor action executor"
```

---

### Task 3: Recorder Schema Normalization

**Files:**
- Modify: `recorder/recorder.py:22-187`
- Test: `tests/test_recorder_schema.py`

**Interfaces:**
- Consumes: input state dict, optional memory state dict, optional visual state dict, optional flow decision dict.
- Produces: normalized JSON rows for `inputs.jsonl`, `memory_state.jsonl`, and `metadata.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_recorder_schema.py`:

```python
from recorder.recorder import build_memory_state_row


def test_build_memory_state_row_includes_frame_timestamp_memory_and_flow():
    row = build_memory_state_row(
        frame_id=7,
        timestamp=12.34,
        memory_state={"hp": 100, "hp_pct": 0.5, "pos_x": 1.0},
        visual_state={"boss_hp": 0.8, "locked": True},
        flow_state={"state": "combat", "state_cn": "战斗中", "reason_cn": "检测到Boss血条"},
    )
    assert row["frame_id"] == 7
    assert row["timestamp"] == 12.34
    assert row["memory"]["hp"] == 100
    assert row["visual"]["boss_hp"] == 0.8
    assert row["flow"]["state_cn"] == "战斗中"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_recorder_schema.py -v`

Expected: FAIL with `ImportError: cannot import name 'build_memory_state_row'`.

- [ ] **Step 3: Add normalization helper**

Add near top of `recorder/recorder.py` after `_save_frame`:

```python
def build_memory_state_row(frame_id: int, timestamp: float,
                           memory_state: dict | None,
                           visual_state: dict | None,
                           flow_state: dict | None) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "source": "recorder",
        "mode_cn": "教师模式" if memory_state else "视觉模式",
        "memory": memory_state or {},
        "visual": visual_state or {},
        "flow": flow_state or {},
    }
```

- [ ] **Step 4: Extend Recorder constructor**

Change `Recorder.__init__` signature from:

```python
def __init__(self, fps: int = config.FPS,
             output_dir: Path | None = None,
             game_state_provider=None):
```

to:

```python
def __init__(self, fps: int = config.FPS,
             output_dir: Path | None = None,
             game_state_provider=None,
             visual_state_provider=None,
             flow_state_provider=None):
```

Inside `__init__`, add:

```python
self._visual_state_provider = visual_state_provider
self._flow_state_provider = flow_state_provider
self._has_visual_state = False
self._has_flow_state = False
```

- [ ] **Step 5: Capture visual and flow state in `_sample_loop`**

Replace buffer tuple creation in `_sample_loop` with logic:

```python
visual_state = None
if self._visual_state_provider is not None:
    try:
        vs = self._visual_state_provider(img)
        if vs is not None:
            visual_state = vs
            self._has_visual_state = True
    except Exception:
        pass

flow_state = None
if self._flow_state_provider is not None:
    try:
        fs = self._flow_state_provider(visual_state, game_state)
        if fs is not None:
            flow_state = fs
            self._has_flow_state = True
    except Exception:
        pass

with self._lock:
    self._buffer.append((img, idx, ts, state, game_state, visual_state, flow_state))
    self.frame_count += 1
```

- [ ] **Step 6: Rename output files**

In `flush()`, write `metadata.json` instead of `meta.json`, and include flags:

```python
"has_memory_state": self._has_game_state,
"has_visual_state": self._has_visual_state,
"has_flow_state": self._has_flow_state,
```

In `_write_batch()`, change:

```python
gs_path = self.output_dir / "game_state.jsonl"
```

to:

```python
gs_path = self.output_dir / "memory_state.jsonl"
```

When writing memory rows, use:

```python
row = build_memory_state_row(idx, ts, game_state, visual_state, flow_state)
gs_log.write(json.dumps(row, ensure_ascii=False) + "\n")
```

Write a row if any of `game_state`, `visual_state`, or `flow_state` is present.

Update final print to:

```python
print(f"[REC] saved: frames/ ({self.frame_count} {ext}) + inputs.jsonl + memory_state.jsonl + metadata.json")
```

- [ ] **Step 7: Run schema test**

Run: `python -m pytest tests/test_recorder_schema.py -v`

Expected: PASS.

- [ ] **Step 8: Run existing smoke import**

Run: `python - <<'PY'
from recorder.recorder import Recorder, build_memory_state_row
print(build_memory_state_row(1, 1.0, None, None, None))
PY`

Expected: prints a dict with `frame_id: 1` and no exception.

- [ ] **Step 9: Commit**

```bash
git add recorder/recorder.py tests/test_recorder_schema.py
git commit -m "feat: normalize recorder state logs"
```

---

### Task 4: Wire State Machine Into Recorder

**Files:**
- Modify: `recorder/run.py:70-83`
- Test: manual import command

**Interfaces:**
- Consumes: `GameState.detect(frame)` and `MemoryReader.read()`.
- Produces: `Recorder(... visual_state_provider=..., flow_state_provider=...)` wiring.

- [ ] **Step 1: Add imports**

In `recorder/run.py`, add near existing imports:

```python
from game_state import GameState
from state_machine import StateSupervisor
```

- [ ] **Step 2: Create providers in `main()`**

After memory provider setup, add:

```python
visual_detector = GameState(resolution=(config.TARGET_WIDTH, config.TARGET_HEIGHT))
supervisor = StateSupervisor()

def visual_state_provider(frame):
    return visual_detector.detect(frame)

def flow_state_provider(visual_state, memory_state):
    decision = supervisor.classify(visual_state, memory_state)
    return {
        "state": decision.state.value,
        "state_cn": decision.state_cn,
        "confidence": decision.confidence,
        "reason_cn": decision.reason_cn,
        "should_release_inputs": decision.should_release_inputs,
    }
```

- [ ] **Step 3: Pass providers into Recorder**

Change:

```python
recorder = Recorder(fps=config.FPS, game_state_provider=game_state_provider)
```

to:

```python
recorder = Recorder(
    fps=config.FPS,
    game_state_provider=game_state_provider,
    visual_state_provider=visual_state_provider,
    flow_state_provider=flow_state_provider,
)
```

- [ ] **Step 4: Update console text**

Change startup text from `录制手柄 + 截图` to:

```python
print("  Game AI Recorder — 录制截图 + 输入 + 内存/视觉状态")
```

If memory opens, print:

```python
print("[REC] 内存读取已启用 → 状态将写入 memory_state.jsonl")
```

If memory is unavailable, print:

```python
print("[REC] 未检测到游戏进程，仅录制画面+输入+视觉状态")
```

- [ ] **Step 5: Run import check**

Run: `python - <<'PY'
from recorder.run import main
print(main.__name__)
PY`

Expected: prints `main` and exits without starting recording.

- [ ] **Step 6: Commit**

```bash
git add recorder/run.py
git commit -m "feat: record visual flow state"
```

---

### Task 5: Live Supervisor Runner

**Files:**
- Create: `run_supervisor.py`
- Test: dry-run command

**Interfaces:**
- Consumes: `Capture`, `GameState`, optional `MemoryReader`, `StateSupervisor`, `ActionExecutor`.
- Produces: live Chinese console decisions and real actions unless `--dry-run` is passed.

- [ ] **Step 1: Create dry-run runnable script**

Create `run_supervisor.py`:

```python
"""状态机监督器入口。

用法：
    python run_supervisor.py --dry-run
    python run_supervisor.py
"""
import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recorder.capture import Capture
from recorder import config
from game_state import GameState
from state_machine import StateSupervisor, FlowState
from control import ActionExecutor

try:
    from memory_reader import MemoryReader
except ImportError:
    MemoryReader = None


def main():
    parser = argparse.ArgumentParser(description="黑夜君临状态机监督器")
    parser.add_argument("--dry-run", action="store_true", help="只打印中文决策，不实际按键")
    args = parser.parse_args()

    dry_run = args.dry_run or config.SUPERVISOR_DRY_RUN
    executor = ActionExecutor(
        dry_run=dry_run,
        confirm_key=config.CONFIRM_KEY,
        lock_on_key=config.LOCK_ON_KEY,
    )
    detector = GameState(resolution=(config.TARGET_WIDTH, config.TARGET_HEIGHT))
    supervisor = StateSupervisor()

    memory_reader = None
    if MemoryReader is not None:
        mr = MemoryReader()
        if mr.open():
            memory_reader = mr
            print("[监督器] 内存读取已启用")
        else:
            print("[监督器] 未检测到游戏进程，仅使用视觉状态")

    capture = Capture(target_fps=max(1, int(1 / config.SUPERVISOR_INTERVAL_SEC)))
    capture.start()
    print(f"[监督器] 启动，dry_run={dry_run}，间隔={config.SUPERVISOR_INTERVAL_SEC}s")

    last_retry_at = 0.0
    try:
        while True:
            frame, frame_id, ts = capture.read()
            if frame is None:
                time.sleep(0.05)
                continue

            visual_state = detector.detect(frame)
            memory_state = memory_reader.read() if memory_reader else None
            decision = supervisor.classify(visual_state, memory_state)
            print(f"[状态机] {decision.state_cn} conf={decision.confidence:.2f}：{decision.reason_cn}")

            if decision.should_release_inputs:
                result = executor.release_all()
                print(result.summary_cn)

            now = time.time()
            if decision.state in (FlowState.DEAD, FlowState.RETRY_PROMPT) and now - last_retry_at >= 2.0:
                print("[规划脑] 决策：执行再战确认")
                first = executor.press_confirm()
                print(first.summary_cn)
                time.sleep(config.RETRY_CONFIRM_DELAY_SEC)
                second = executor.press_confirm()
                print(second.summary_cn)
                last_retry_at = now
            elif decision.state == FlowState.COMBAT:
                locked = False
                if visual_state:
                    locked = bool(visual_state.get("locked") or visual_state.get("locked_on"))
                if memory_state:
                    locked = locked or bool(memory_state.get("locked_on"))
                if not locked:
                    print("[规划脑] 决策：尝试锁定目标")
                    result = executor.lock_on()
                    print(result.summary_cn)
                else:
                    print("[规划脑] 战斗中且已锁定，等待战斗脑接管")

            time.sleep(config.SUPERVISOR_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n[监督器] 退出，释放按键")
        print(executor.release_all().summary_cn)
    finally:
        capture.stop()
        if memory_reader:
            memory_reader.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run help check**

Run: `python run_supervisor.py --help`

Expected: prints argparse help with `--dry-run`.

- [ ] **Step 3: Do not run live loop on company computer**

Do not run `python run_supervisor.py` or `python run_supervisor.py --dry-run` during company-computer implementation because it starts screen capture.

- [ ] **Step 4: Commit**

```bash
git add run_supervisor.py
git commit -m "feat: add live state supervisor runner"
```

---

### Task 6: Update Docs For New Workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/memory_state_spec.md`
- Modify: `docs/decision_trace_design.md`

**Interfaces:**
- Consumes: implemented commands and output filenames.
- Produces: clear Chinese instructions for recording and supervisor usage.

- [ ] **Step 1: Update README recording section**

In `README.md`, replace the recording output sentence with:

```markdown
录制数据保存在 `data/session_*/` 目录：

- `frames/`：720p JPEG 截图
- `inputs.jsonl`：每帧输入
- `memory_state.jsonl`：内存/视觉/状态机标签（有多少写多少）
- `metadata.json`：本次录制配置
```

- [ ] **Step 2: Add supervisor usage to README**

Add under quick start:

```markdown
### 状态机监督器

```bash
# 安全预览：只打印中文决策，不按键
python run_supervisor.py --dry-run

# 真实执行：死亡后再战、战斗中尝试锁定
python run_supervisor.py
```

公司电脑不要运行监督器；只在个人电脑、PVE/离线/练习 Mod 环境中使用。
```

- [ ] **Step 3: Update docs wording**

In `docs/memory_state_spec.md`, ensure output filename is `memory_state.jsonl` and metadata filename is `metadata.json`.

In `docs/decision_trace_design.md`, add one line:

```markdown
第一版监督器日志先输出到终端；后续再落盘到 `planner_trace_*.jsonl`。
```

- [ ] **Step 4: Run markdown read check**

Run: `python - <<'PY'
from pathlib import Path
for p in [Path('README.md'), Path('docs/memory_state_spec.md'), Path('docs/decision_trace_design.md')]:
    text = p.read_text(encoding='utf-8')
    assert 'memory_state.jsonl' in text or p.name == 'decision_trace_design.md'
print('docs ok')
PY`

Expected: prints `docs ok`.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/memory_state_spec.md docs/decision_trace_design.md
git commit -m "docs: document recorder and supervisor workflow"
```

---

### Task 7: Final Verification

**Files:**
- No new files.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: evidence that offline tests pass and live-only commands are documented but not run.

- [ ] **Step 1: Run unit tests**

Run: `python -m pytest tests/test_state_machine.py tests/test_action_executor.py tests/test_recorder_schema.py -v`

Expected: all PASS.

- [ ] **Step 2: Run supervisor help only**

Run: `python run_supervisor.py --help`

Expected: argparse help prints successfully. Do not run the live loop.

- [ ] **Step 3: Run recorder import check**

Run: `python - <<'PY'
from recorder.recorder import Recorder
from state_machine import StateSupervisor
from control import ActionExecutor
print('imports ok')
PY`

Expected: prints `imports ok`.

- [ ] **Step 4: Check git status**

Run: `git status --short`

Expected: only intended files modified/created.

- [ ] **Step 5: Final note**

Report:

```text
已完成离线可测的录制+状态机基础：录制器输出 memory_state.jsonl / metadata.json，状态机可分类死亡/战斗/未知，监督器支持 dry-run 和真实按键模式。公司电脑未运行游戏、CE、Mod 或 live supervisor loop。
```
