from recorder.episode_splitter import EpisodeSplitter
from recorder.reward import RewardCalculator


def test_splitter_finishes_on_reward_win():
    calc = RewardCalculator(time_penalty=0.0)
    calc.observe(1, 1.0, {"hp_pct": 1.0, "boss_hp_pct": 0.1})
    reward = calc.observe(2, 2.0, {"hp_pct": 1.0, "boss_hp_pct": 0.0})

    finish, reason, result = EpisodeSplitter().should_finish(1.0, 2.0, reward_result=reward)

    assert finish is True
    assert reason == "boss_dead"
    assert result == "win"


def test_splitter_finishes_on_state_death_without_reward():
    finish, reason, result = EpisodeSplitter().should_finish(
        1.0, 2.0, game_state={"hp_pct": 0.0, "boss_hp_pct": 0.8}
    )

    assert finish is True
    assert reason == "player_dead"
    assert result == "death"


def test_splitter_finishes_on_timeout():
    finish, reason, result = EpisodeSplitter(timeout_sec=10.0).should_finish(1.0, 11.1)

    assert finish is True
    assert reason == "timeout"
    assert result == "timeout"
