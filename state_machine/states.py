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
