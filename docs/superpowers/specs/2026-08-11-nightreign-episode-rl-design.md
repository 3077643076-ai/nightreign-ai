# Nightreign AI：Episode 录制与 Boss 训练规划

日期：2026-08-11

## 目标

当前目标不是先做通用游戏 AI，而是先做固定训练场 Boss 的视觉强化学习闭环：

1. 固定 Boss、固定角色、固定武器、固定难度、固定开局。
2. 先追求稳定击杀，再优化少掉血、无伤、速杀。
3. 用人类键鼠录制作为示范和离线经验，后续接 RL 微调。
4. 普通 Boss 和机制 Boss 分开处理，不强行混入同一个泛化任务。

## 参考项目定位

### RL_For_RPG

主要参考真实游戏 RL 工程落地方式：

- 截图作为视觉输入。
- 将真实游戏封装成环境。
- 设计离散动作空间。
- 根据血量、击杀、死亡等反馈计算 reward。
- 形成 `state -> action -> reward -> next_state -> done` 闭环。

它和本项目更接近，适合作为工程参考。

### ViZDoom

主要参考标准 RL 环境设计方式：

- Gymnasium 风格接口。
- episode、scenario、reward、done 的标准拆分。
- 用固定场景做可复现实验。
- 论文中可以参考其“视觉输入 + 强化学习环境”的表达方式。

它不直接迁移游戏逻辑，但适合参考实验设计和接口标准。

## 当前判断

现有项目已经有：

- 截图录制。
- 手柄/键鼠输入记录基础。
- HUD / HP / boss HP 等感知模块。
- 行为克隆训练脚本。
- 简单控制执行模块。
- “战斗脑 + 规划脑”的整体方向。

后续最该补的是 Episode 录制层和 RL 环境层，而不是直接先改模型。

## 录制方向

现有录制方式更接近连续录制。后续需要升级为按局保存：

```text
data/episodes/
  ep_000001/
    frames/
    inputs.jsonl
    states.jsonl
    rewards.jsonl
    meta.json
```

每一局 episode 都应该独立保存，方便后续筛选、训练和复现实验。

### 每局 meta 信息

建议 `meta.json` 至少包含：

```json
{
  "episode_id": "ep_000001",
  "boss_id": "tree_boss",
  "boss_type": "mechanic",
  "difficulty": "normal",
  "control": "keyboard_mouse",
  "weapon": "special_weapon",
  "required_action": "weapon_art",
  "result": "win",
  "duration_sec": 83.4,
  "frame_count": 1248,
  "final_player_hp": 1.0,
  "final_boss_hp": 0.0,
  "end_reason": "boss_dead"
}
```

## Boss 标记方案

第一版不建议依赖 OCR 自动识别 Boss。更稳的方案是：

1. 录制前手动选择 `boss_id`。
2. 这一批 episode 自动继承该 Boss 标签。
3. OCR 后续只作为校验，不作为唯一真值。

原因：OCR 可能受字体、特效、背景和显示时机影响，识别错会污染整局数据。

推荐优先级：

```text
手动配置 > 训练场/脚本选择记录 > 画面特征识别 > OCR
```

## Boss 类型划分

### 标准战斗 Boss

特点：

- 不依赖特殊武器或机制。
- 主要考察看招、躲避、接近、输出窗口。
- 适合用于跨 Boss 泛化或微调实验。

训练目标：

```text
稳定击杀 -> 少掉血 -> 无伤 -> 速杀
```

### 机制 Boss

特点：

- 需要指定武器、战技、道具或机制。
- 例如系统发一把特殊武器，只能用该武器战技才能有效击杀。
- 不适合直接混入普通 Boss 泛化训练。

处理方式：

```text
规划脑提供机制先验：这个 Boss 必须使用指定战技。
战斗脑 / RL 只学习什么时候安全、高效地使用战技。
```

机制 Boss 的配置示例：

```json
{
  "boss_id": "tree_boss",
  "boss_type": "mechanic",
  "required_weapon": "special_storm_weapon",
  "required_action": "weapon_art",
  "allow_general_policy": false
}
```

## 难度策略

如果高难只增加 Boss 血量和伤害，不改变招式、AI、阶段或机制，那么当前目标下不需要随机难度。

理由：

- 无伤目标下，Boss 伤害增加没有训练信息。
- Boss 血量增加只会拉长战斗，不改变核心策略。
- 真正有价值的变化来自不同 Boss 的招式、体型、距离和后摇。

因此当前推荐：

```text
固定难度，优先换 Boss。
```

除非高难会改变 Boss 行为，否则不把跨难度泛化作为第一阶段目标。

## 数据使用方式

人类录制数据按质量分级：

### A 类：高质量 / 打赢 / 流程顺

用途：

- 行为克隆训练。
- 作为 RL 初始策略参考。

### B 类：差点赢 / 有失误但流程完整

用途：

- 离线 RL 经验。
- 可选加入 BC，但需要谨慎。

### C 类：很快死亡 / 明显乱按 / 卡墙

用途：

- 不进入 BC。
- 可作为 RL 负样本或 reward 调试样本。

核心原则：

```text
赢的局当老师，输的局当经验，乱的局当负样本。
```

## Reward 初版设计

目标是无伤速杀，因此 reward 不应只奖励胜利。

建议：

```text
Boss 掉血：正奖励
击杀 Boss：大正奖励
自己掉血：大惩罚
死亡：超大惩罚
时间流逝：小惩罚
长时间未造成伤害：小惩罚
空挥 / 无效攻击：惩罚
无伤击杀：额外奖励
```

机制 Boss 需要额外处理：

```text
正确使用指定战技并造成有效进展：正奖励
普通攻击无效或浪费输出窗口：惩罚或不给奖励
```

## 动作空间方向

当前倾向切换到键鼠录制。键鼠对离散动作更友好，但鼠标视角控制复杂。

第一版建议尽量固定锁定 Boss，减少鼠标学习压力。

标准 Boss 初版动作空间：

```text
0 = 无动作
1 = 前进
2 = 后退
3 = 左移
4 = 右移
5 = 翻滚
6 = 轻攻击
7 = 重攻击
8 = 锁定
9 = 喝药
10 = 战技
```

机制 Boss 可以单独缩小动作空间：

```text
移动
翻滚
锁定
战技
后撤
喝药
```

如果能稳定锁定 Boss，鼠标移动第一版可以不进入动作空间。

## Episode 自动切分

不需要手动分割每局。训练场 mod 如果能死亡后点“再战”马上重开，应做自动切局。

可用切局信号：

1. 玩家死亡检测。
2. Boss HP 归零。
3. 再战 / 确认界面出现。
4. 玩家 HP 或 Boss HP 重置。
5. 超时兜底。

流程：

```text
开始录制
循环保存 frame/input/state
检测 death/win/timeout
保存当前 episode
自动按确认/再战
等待新一局开始
开启下一个 episode
```

## RL 环境层方向

后续需要新增类似 `NightreignEnv` 的环境层：

```python
class NightreignEnv:
    def reset(self):
        ...

    def step(self, action):
        ...

    def get_observation(self):
        ...

    def compute_reward(self):
        ...

    def is_done(self):
        ...
```

它负责把真实游戏交互包装成标准 RL 闭环。

## 明天优先做的事情

1. 确定键鼠录制输入格式。
2. 设计 `episode` 目录结构和 `meta.json` 字段。
3. 给录制脚本增加录制前 `boss_id` / `boss_type` / `difficulty` 配置。
4. 增加自动切局逻辑：death、boss_dead、retry、timeout。
5. 先不做 OCR Boss 识别，只保留将来校验接口。
6. 跑一个最小测试：连续录 2-3 局，每局自动保存成独立 episode。

## 暂不做

- 不做跨难度随机化。
- 不做通用 Boss 识别模型。
- 不让 OCR 作为 Boss 标签真值。
- 不把机制 Boss 混入普通 Boss 泛化训练。
- 不一开始做完整 PPO/DQN 训练，先把数据与环境闭环跑通。

## 论文表达方向

可以写成：

```text
本文采用固定训练场 Boss 场景，构建基于视觉输入的游戏强化学习环境。
系统首先利用人类键鼠演示进行行为克隆预训练，随后通过基于血量变化、死亡、击杀和时间成本的奖励函数进行强化学习微调。
对于机制约束型 Boss，本文不将其视为普通跨 Boss 泛化样本，而是通过规则配置提供机制先验，使强化学习策略专注于战斗执行时机优化。
```

## 周末最小验证清单

目标不是训练模型，而是确认录制系统输出正确。

### 启动前

在 PowerShell 或 cmd 里设置 Boss 标签：

```bash
set NIGHTREIGN_BOSS_ID=training_boss
set NIGHTREIGN_BOSS_TYPE=standard
set NIGHTREIGN_DIFFICULTY=normal
set NIGHTREIGN_WEAPON=greatsword
```

### 录制流程

1. 启动游戏和训练场。
2. 运行 `python record.py`。
3. 按 F8 开始录制。
4. 用键鼠打 30-60 秒。
5. 按 F9 停止录制。
6. 打开最新 `data/episode_*` 目录。

### 必须检查的输出

- `frames/` 有 jpg 图片。
- `inputs.jsonl` 每行都有 `gamepad`、`keyboard`、`mouse`。
- 如果内存读取成功，`game_state.jsonl` 有 `hp_pct`、`fp_pct`、`stamina_pct`。
- `rewards.jsonl` 存在，并且每行有 `reward`、`events`、`done`。
- `meta.json` 有 `boss_id`、`boss_type`、`difficulty`、`weapon`、`frame_count`、`duration_sec`。

### 成功标准

只要能证明“画面、键鼠、状态、奖励、Boss 标签”在同一帧号附近对齐，就算第一轮成功。
