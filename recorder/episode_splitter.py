class EpisodeSplitter:
    def __init__(self, timeout_sec: float | None = None):
        self.timeout_sec = timeout_sec

    def should_finish(self, start_timestamp: float | None, timestamp: float,
                      reward_result=None, game_state: dict | None = None) -> tuple[bool, str, str]:
        if reward_result is not None and reward_result.done:
            if reward_result.result == "win":
                return True, "boss_dead", "win"
            if reward_result.result == "death":
                return True, "player_dead", "death"

        if game_state is not None:
            if self._is_dead(game_state):
                return True, "player_dead", "death"
            if self._boss_is_dead(game_state):
                return True, "boss_dead", "win"

        if self.timeout_sec is not None and start_timestamp is not None:
            if timestamp - start_timestamp >= self.timeout_sec:
                return True, "timeout", "timeout"

        return False, "", ""

    def _is_dead(self, game_state: dict) -> bool:
        if game_state.get("is_dead") is True:
            return True
        hp_pct = game_state.get("hp_pct")
        return hp_pct is not None and hp_pct >= 0 and hp_pct <= 0.0

    def _boss_is_dead(self, game_state: dict) -> bool:
        boss_hp_pct = game_state.get("boss_hp_pct")
        return boss_hp_pct is not None and boss_hp_pct >= 0 and boss_hp_pct <= 0.0
