# 内存状态采集规范

## 目标

内存读取不是项目的“作弊核心”，而是训练阶段的老师：

```text
画面 + 输入 + 内存真值
        ↓
奖励计算 / 状态标注 / 视觉状态估计训练 / 调试验证
        ↓
逐步减少对内存的依赖
```

本项目只在 PVE、离线、练习 Mod 或个人研究环境中采集内存状态，不用于联机对抗。

## 数据保存位置

录制目录统一采用：

```text
data/
  session_001/
    frames/
      000001.jpg
      000002.jpg
    inputs.jsonl
    memory_state.jsonl
    decision_trace.jsonl
    metadata.json
```

其中：

- `frames/`：截图帧
- `inputs.jsonl`：手柄/键鼠输入
- `memory_state.jsonl`：每帧内存状态真值
- `decision_trace.jsonl`：可选，推理/训练时的决策轨迹
- `metadata.json`：本局配置，例如角色、武器、Boss、Mod、分辨率、fps

## 每帧记录格式

字段名建议使用英文，方便代码读取；中文解释字段用于人工查看和论文展示。

```json
{
  "frame_id": 18342,
  "timestamp": 123.45,
  "source": "memory_reader",
  "mode_cn": "教师模式",
  "player": {
    "hp": 812,
    "max_hp": 1200,
    "hp_pct": 0.6767,
    "fp": 80,
    "max_fp": 100,
    "fp_pct": 0.8,
    "stamina": 94,
    "max_stamina": 130,
    "stamina_pct": 0.7231,
    "pos_x": 124.52,
    "pos_y": 31.08,
    "pos_z": -502.77,
    "yaw": 182.4,
    "anim_id": 30210,
    "action_cn": "向左翻滚"
  },
  "target": {
    "locked_on": true,
    "name_cn": "夜王",
    "hp_pct": 0.61,
    "anim_id": 80120,
    "action_cn": "蓄力攻击",
    "distance": 4.2
  },
  "flow": {
    "combat_state": "combat",
    "combat_state_cn": "战斗中",
    "in_menu": false,
    "is_dead": false,
    "retry_available": false
  },
  "location": {
    "area_id": null,
    "area_name_cn": "训练场夜王区域",
    "nearest_waypoint": null,
    "target_waypoint": null
  }
}
```

## 阶段一：训练场 / Boss 房必须读取

这些字段服务于纯 RL 战斗脑和奖励函数。

| 字段 | 中文含义 | 当前状态 | 用途 |
|---|---|---|---|
| `hp`, `max_hp`, `hp_pct` | 玩家血量 | 已有基础 | 受伤惩罚、死亡判断、喝药收益 |
| `fp`, `max_fp`, `fp_pct` | 玩家蓝量 | 已有基础 | 战技资源判断 |
| `stamina`, `max_stamina`, `stamina_pct` | 玩家耐力 | 已有基础 | 乱滚、低耐力攻击惩罚 |
| `pos_x`, `pos_y`, `pos_z` | 玩家坐标 | 已有基础 | 距离、位移、卡住检测 |
| `player_yaw` | 玩家朝向 | 待补 | 是否面向 Boss、后撤/绕圈判断 |
| `player_anim_id` | 玩家动画 ID | 已有基础 | 动作分割、死亡/处决/喝药识别 |
| `locked_on` | 是否锁定 | 待补 | 战斗开始、锁定稳定性 |
| `target_hp_pct` | Boss 血量 | 待补 | 造成伤害奖励 |
| `target_anim_id` | Boss 动画 ID | 有雏形，需验证 | 攻击前摇、弹反窗口、危险状态 |
| `distance_to_target` | 与目标距离 | 待补 | 接近、后撤、攻击距离判断 |
| `combat_state` | 战斗状态 | 待统一 | 状态机切换 |
| `is_dead` | 是否死亡 | 可由 HP/动画推导 | 重开流程 |
| `retry_available` | 是否出现再战确认 | 待视觉/状态机识别 | 死亡后自动再战 |

## 阶段二：爬塔 / Boss Rush 需要读取

这些字段服务于连续战斗和装备规划脑。

| 字段 | 中文含义 | 用途 |
|---|---|---|
| `current_stage` | 当前爬塔层数/场次 | 统计连续战斗表现 |
| `enemy_name_cn` | 当前敌人/Boss 名称 | 装备和打法分析 |
| `current_weapon_cn` | 当前武器 | 固定单大剑流派确认 |
| `current_relics` | 当前遗物/词条 | 装备规划脑输入 |
| `available_rewards` | 可选奖励 | 装备规划脑打分 |
| `chosen_reward` | 已选奖励 | 记录规划脑决策 |
| `stage_result` | 本场结果 | 胜负、耗时、死亡原因 |
| `damage_dealt_total` | 总造成伤害 | 评估装备收益 |
| `damage_taken_total` | 总受到伤害 | 评估生存收益 |

如果这些字段无法直接从内存读取，第一版允许用 OCR、UI 检测或人工配置补齐。

## 阶段三：正常对局地图规划需要读取

这是后续扩展，不纳入毕设最低完成标准，但从录制阶段就应该保留字段位置。

| 字段 | 中文含义 | 用途 |
|---|---|---|
| `area_id` | 当前区域 ID | 区分地图区域、Boss 场、训练场 |
| `area_name_cn` | 当前地点中文名 | 中文日志、论文展示 |
| `pos_x/y/z` | 玩家坐标 | 路线规划核心 |
| `player_yaw` | 玩家朝向 | 是否朝向目标点 |
| `nearest_waypoint` | 最近地图节点 | 坐标转语义地点 |
| `target_waypoint` | 当前目标节点 | 规划脑决定去哪 |
| `distance_to_waypoint` | 到目标点距离 | 到达/走偏判断 |
| `stuck_score` | 卡住分数 | 撞墙、原地打转检测 |
| `night_circle_state` | 黑夜圈状态 | 正常局路线优先级 |
| `in_map` | 是否打开地图 | 地图定位/视觉识别 |
| `in_inventory` | 是否打开背包 | 装备 OCR / 规划脑输入 |
| `post_game_path` | 赛后行动路径 | 路线复盘与评估 |

## Memory Teacher → Vision Student

每个新阶段开始时，先允许内存读取作为老师。

```text
Teacher Mode：画面 + 输入 + 内存真值
Hybrid Mode：画面预测 + 内存校验
Student Mode：主要依赖画面 / HUD / OCR / 小地图 / 历史记忆
```

可逐步关闭的字段：

- HP / FP / 耐力：可由 HUD 视觉估计替代
- Boss HP：可由血条检测替代
- 锁定状态：可由锁定标记和镜头行为替代
- 距离：可由敌人大小、透视和坐标真值蒸馏得到
- 敌人动作：可由 Boss 动作分类器替代

不建议过早关闭的字段：

- 精确世界坐标
- 地图区域 ID
- 路线节点误差
- 赛后路径复盘

## 中文字典

需要维护若干字典，把数字 ID 翻译成人话。

```text
anim_id → 动作中文名
boss_id → Boss 中文名
enemy_id → 敌人中文名
weapon_id → 武器中文名
relic_id → 遗物/词条中文名
area_id → 地点中文名
waypoint_id → 地图节点中文名
```

示例：

```json
{
  "30210": "向左翻滚",
  "30211": "向右翻滚",
  "80120": "Boss 蓄力攻击",
  "90001": "处决窗口"
}
```

## 日志要求

- 字段名使用英文，避免后续 Python 分析困难。
- `*_cn` 字段必须提供中文说明。
- 内存读取失败时记录 `null`，不要伪造数据。
- 状态摘要必须能让人直接看懂：例如“玩家血量 67%，Boss 正在蓄力，距离 4.2 米”。
- 公司电脑不运行内存读取，只维护文档和离线代码。
