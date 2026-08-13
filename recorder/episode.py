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
