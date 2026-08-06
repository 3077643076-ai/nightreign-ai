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
