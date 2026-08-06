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
