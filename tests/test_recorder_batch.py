import json

import numpy as np

from recorder.boss_config import BossConfig
from recorder.episode import EpisodeTracker
from recorder.episode_splitter import EpisodeSplitter
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


def test_auto_episode_rolls_to_next_directory_after_death(tmp_path):
    recorder = Recorder(
        fps=15,
        output_dir=tmp_path,
        input_provider=FakeInputProvider(),
        boss_config=BossConfig(boss_id="test_boss"),
        reward_calculator=RewardCalculator(time_penalty=0.0),
        episode_splitter=EpisodeSplitter(),
        auto_episode=True,
    )
    recorder._start_new_episode(timestamp=1.0)
    first_dir = recorder.output_dir

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    gamepad_state = {"buttons": {}, "axes": {}}
    keyboard_mouse_state = {"keys": {}, "mouse_buttons": {}, "mouse_delta": {"dx": 0, "dy": 0, "wheel": 0}}

    recorder._write_batch([
        (frame, 1, 1.0, gamepad_state, keyboard_mouse_state, {"hp_pct": 0.5, "boss_hp_pct": 0.8}),
        (frame, 2, 2.0, gamepad_state, keyboard_mouse_state, {"hp_pct": 0.0, "boss_hp_pct": 0.8}),
    ])

    first_meta = json.loads((first_dir / "meta.json").read_text(encoding="utf-8"))
    assert first_meta["result"] == "death"
    assert first_meta["end_reason"] == "player_dead"
    assert recorder.output_dir != first_dir
    assert recorder.output_dir.name.endswith("0002")
