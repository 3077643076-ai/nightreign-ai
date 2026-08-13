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
