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

    def reset(self):
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
