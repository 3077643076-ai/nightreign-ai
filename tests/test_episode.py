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
