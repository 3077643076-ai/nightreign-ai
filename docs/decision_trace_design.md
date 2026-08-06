# 中文决策轨迹设计

## 目标

决策轨迹系统用于回答三个问题：

```text
AI 看到了什么？
AI 为什么选择这个动作？
这个动作执行后奖励是多少？
```

本项目不记录所谓“真实思维链”，而是记录可验证、可回放、可统计的决策依据：状态、候选动作、分数、选择、执行结果和奖励分项。

## 日志文件

```text
logs/
  combat_trace_2026-08-06.jsonl
  planner_trace_2026-08-06.jsonl
  reward_trace_2026-08-06.jsonl
  session_summary_2026-08-06.json
```

要求：

- 一行一个事件，使用 jsonl。
- 字段名英文，中文展示字段使用 `_cn` 后缀。
- 终端实时输出中文摘要。
- 训练后生成中文统计报告。

## 战斗脑日志

用于记录 RL 战斗脑每次动作选择。

```json
{
  "event_type": "combat_decision",
  "brain": "combat",
  "brain_cn": "战斗脑",
  "timestamp": 123.45,
  "frame_id": 18342,
  "state_summary_cn": "玩家血量67%，耐力72%，Boss血量61%，Boss正在蓄力攻击，距离4.2米",
  "planner_intent": "safe_defense",
  "planner_intent_cn": "稳健防守",
  "allowed_actions": ["roll_left", "roll_right", "parry", "retreat"],
  "blocked_actions": ["heavy_attack", "heal"],
  "candidates": [
    {
      "action": "roll_left",
      "action_cn": "向左翻滚",
      "score": 0.82,
      "reason_cn": "Boss正在蓄力，左侧空间安全，规避收益最高"
    },
    {
      "action": "parry",
      "action_cn": "弹反",
      "score": 0.61,
      "reason_cn": "可能存在弹反窗口，但当前识别置信度不够高"
    },
    {
      "action": "retreat",
      "action_cn": "后撤",
      "score": 0.33,
      "reason_cn": "可以拉开距离，但会降低输出机会"
    }
  ],
  "chosen_action": "roll_left",
  "chosen_action_cn": "向左翻滚",
  "decision_reason_cn": "向左翻滚得分最高，且符合规划脑的稳健防守意图"
}
```

终端摘要：

```text
[战斗脑] 玩家血量67%，耐力72%，Boss血量61%，Boss正在蓄力攻击，距离4.2米
[规划意图] 稳健防守，可选：向左翻滚/向右翻滚/弹反/后撤
[候选动作] 向左翻滚=0.82，弹反=0.61，后撤=0.33
[最终选择] 向左翻滚：规避收益最高
```

## 规划脑日志

用于记录低频指挥、状态机流程和装备选择。

### 流程控制事件

```json
{
  "event_type": "planner_flow",
  "brain": "planner",
  "brain_cn": "规划脑",
  "timestamp": 456.78,
  "state": "dead",
  "state_cn": "死亡界面",
  "decision": "retry",
  "decision_cn": "执行再战流程",
  "steps_cn": [
    "等待0.5秒，避免误触",
    "按下A/F确认再战",
    "等待确认框消失",
    "再次按下A/F",
    "等待加载进入战斗"
  ],
  "reason_cn": "检测到玩家死亡且再战按钮可用，当前不允许战斗脑继续输出动作"
}
```

终端摘要：

```text
[规划脑] 当前状态：死亡界面
[规划脑] 决策：执行再战流程
[动作执行] 等待0.5秒 → 按A/F → 再确认 → 等待加载
```

### 战斗意图事件

```json
{
  "event_type": "planner_intent",
  "brain": "planner",
  "brain_cn": "规划脑",
  "timestamp": 789.01,
  "intent": "greedy_attack",
  "intent_cn": "允许贪刀",
  "allowed_actions": ["light_attack", "heavy_attack", "weapon_art", "riposte"],
  "blocked_actions": ["heal"],
  "reason_cn": "Boss刚结束大动作进入硬直，玩家耐力充足，适合抢输出"
}
```

### 装备选择事件

```json
{
  "event_type": "planner_equipment",
  "brain": "planner",
  "brain_cn": "规划脑",
  "phase": "tower_reward_selection",
  "phase_cn": "爬塔奖励选择",
  "timestamp": 1000.12,
  "state_summary_cn": "当前流派：初始单大剑，目标：提高近战输出和生存能力",
  "options": [
    {
      "item": "战技伤害+12%",
      "score": 0.91,
      "reason_cn": "符合大剑战技流，直接提高主要输出"
    },
    {
      "item": "最大生命+8%",
      "score": 0.63,
      "reason_cn": "提高容错率，但输出收益低于战技伤害"
    },
    {
      "item": "智力+3",
      "score": -0.4,
      "reason_cn": "当前流派不依赖法术，收益很低"
    }
  ],
  "chosen_item": "战技伤害+12%",
  "decision_reason_cn": "该词条最符合当前大剑输出路线，得分最高"
}
```

## 奖励日志

用于记录动作执行后的反馈分数。

```json
{
  "event_type": "reward_breakdown",
  "brain": "combat",
  "brain_cn": "战斗脑",
  "timestamp": 124.20,
  "frame_id_start": 18342,
  "frame_id_end": 18368,
  "action": "roll_left",
  "action_cn": "向左翻滚",
  "reward": {
    "damage_dealt": {
      "name_cn": "造成伤害",
      "score": 0.0
    },
    "damage_taken": {
      "name_cn": "受到伤害",
      "score": 0.0
    },
    "survived_attack": {
      "name_cn": "成功规避攻击",
      "score": 1.0
    },
    "stamina_cost": {
      "name_cn": "耐力消耗",
      "score": -0.1
    },
    "total": {
      "name_cn": "总奖励",
      "score": 0.9
    }
  },
  "result_summary_cn": "向左翻滚成功躲过攻击，未造成伤害，消耗少量耐力，总奖励+0.9"
}
```

终端摘要：

```text
[奖励] 向左翻滚：成功规避攻击 +1.0，耐力消耗 -0.1，总奖励 +0.9
```

## 训练后统计报告

每场战斗结束后生成摘要：

```json
{
  "session_id": "session_001",
  "boss_cn": "夜王",
  "result_cn": "失败",
  "duration_sec": 184.2,
  "death_reason_cn": "低耐力时贪重攻击，被Boss连击击杀",
  "metrics": {
    "damage_dealt_total": 0.72,
    "damage_taken_total": 1.0,
    "roll_count": 83,
    "roll_success_rate": 0.68,
    "parry_count": 12,
    "parry_success_rate": 0.25,
    "heal_count": 4,
    "bad_heal_count": 2,
    "empty_attack_count": 19
  },
  "summary_cn": "本局主要问题是空挥和低耐力贪刀，建议提高低耐力攻击惩罚，并增加Boss硬直判断权重。"
}
```

中文报告示例：

```text
第12场 夜王战斗：失败
- 存活时间：184.2秒
- 死亡原因：低耐力时贪重攻击，被Boss连击击杀
- 翻滚：83次，成功率68%
- 弹反：12次，成功率25%
- 喝药：4次，其中2次时机错误
- 空挥：19次
- 建议：提高低耐力攻击惩罚，降低远距离重攻击评分
```

## 调奖励函数时重点看什么

优先观察：

- 是否一直乱滚
- 是否贴脸不输出
- 是否远离 Boss 太久
- 是否残血不喝药
- 是否 Boss 抬手还贪刀
- 是否频繁空挥
- 是否低耐力攻击
- 是否弹反窗口判断错误
- 是否死亡后状态机能稳定重开

## 中文要求

- 所有终端输出必须中文。
- 所有 `*_cn` 字段必须是自然中文，不要机械翻译。
- 数值保留 2 位小数或百分比，方便人工看。
- 每个动作选择必须有一句中文原因。
- 每个奖励分项必须有中文名称。
