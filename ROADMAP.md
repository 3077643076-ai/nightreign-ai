# Game AI Agent — 训练 AI 打游戏

## 目标

训练一个能当队友/单通的通用游戏 AI，从黑环（黑夜君临）起步。
当前角色：追踪者，物理战技弹反处决流。

## 核心架构：双脑分离

黑环决策天然分为两层，用不同方案解决：

```
┌─────────────────────┐    ┌─────────────────────┐
│  战斗脑 (帧级)       │    │  规划脑 (秒/分钟级)    │
│  CNN 行为克隆        │    │  OCR + 规则引擎 + LLM │
│  画面 → 翻滚/弹反/攻击 │    │  文字 → 选装备/路线    │
│  BC 能学好 ✅        │    │  BC 学不了, 用规则 ❌→✅ │
└─────────────────────┘    └─────────────────────┘
```

## 技术栈

| 用途 | 工具 |
|------|------|
| 截图 | DXCam |
| 手柄录制 | `inputs` 库 (XInput) |
| 虚拟手柄 | vgamepad |
| 深度学习 | PyTorch + CUDA |
| 目标检测 | YOLOv8 |
| RL 框架 | stable-baselines3 |

## 项目结构

```
game-ai-agent/
├── recorder/          # 录制工具（截图 + 手柄）
├── preprocess/        # 数据预处理、降质、清洗
├── ocr/               # OCR 文字提取 + 装备词条规则引擎
├── models/            # 模型定义（CNN/IDM/LSTM）
├── train/             # 训练脚本
├── inference/         # 推理部署
├── vgamepad/          # 虚拟手柄控制
├── data/              # 原始数据（gitignore）
└── checkpoints/       # 模型权重（gitignore）
```

## 阶段规划

### Phase 0：录制工具 ✅ 已完成

- [x] 录制脚本：DXCam 截图 + inputs 读手柄 → 720p/15fps/JPEG + inputs.jsonl
- [x] 验证数据：2 局 60756 帧 68 分钟，14.8fps，按键+摇杆+扳机正常
- [x] 热键启动：F8 开始 / F9 停止 / Ctrl+C 退出
- [x] 一键启动：run.bat（自动提权+依赖检查）
- [x] LT/RT 扳机二值化（>=0.5=按下）
- [x] 论文已下载：CS:GO BC + Dark Souls Combat → X:\dev\ai_books\

### Phase 1：战斗底座（黑环）🔄 进行中

目标：追踪者物理战技流，弹反处决，10-15 把自己数据 + 大佬视频。

- [ ] 录 10-15 把追踪者物理战技流（普通难度）
- [ ] 录 X 把深夜难度（高难决策差异）
- [ ] 预处理：2K 降质 720p，清洗死亡/拔线/菜单退出片段
- [ ] 拔线处理：保留死前有效战斗，裁掉退出部分
- [ ] 训练基础 CNN：画面 → 手柄按键 + 摇杆
- [ ] 评估：走路+打怪+翻滚+弹反+喝血+处决
- [ ] 喂大佬追踪者弹反视频，VPT 伪标注扩充

**当前进度**：2 局 / 目标 15+ 局，~60K 帧已录，X 盘剩余 300 GB。

### Phase 1.5：局外装备管理 — OCR + 规则引擎

AI 需要理解装备词条才能选装备，BC 做不到，用规则引擎解决。

- [ ] OCR 模块：识别结算界面遗物/装备词条文字
- [ ] 词条评分：战技攻击力/近战攻击力/满血攻击力 = ★★★ 留
- [ ] 废品识别：魔力/智力/无用词条/负面词条 = 卖
- [ ] 商店筛选：有攻击词条 → 抓
- [ ] 终端补属性：根据现有终端补对应属性攻击
- [ ] 可选：本地小 LLM 评估稀有词条组合

### Phase 2：VPT 视频消化

- [ ] 用自己数据训"动作识别器"（画面→按键）
- [ ] 用识别器翻译大佬弹反视频 → 伪标注按键序列
- [ ] 伪标注数据扩充训练集
- [ ] 法师塔7种解法 → 各录视频 → 伪标注 → AI 全学会
- [ ] 特殊地形路线 → 多录几次固定路线

### Phase 3：RL 自我博弈

- [ ] 定义奖励函数（通关速度+输出效率+不死+处决次数）
- [ ] 高难度 RL 微调
- [ ] AI 自主优化

### Phase 4：队友模式

- [ ] 录联机数据
- [ ] 训练队友决策层（救人、不抢装备、跟随）
- [ ] 同底座切换单通/队友模式

### Phase 5：迁移到其他游戏

- [ ] 怪猎 Wilds（同 3D 底座，微调）
- [ ] 死亡细胞（2D 横版底座，重训视觉层）
- [ ] MC（Mineflayer + 现成方案）

## 路线规划规则（普通难度，追踪者物理流）

从用户习惯中提炼，用于规划脑规则引擎：

```
落地阶段:
  P0: 落地 → 最近据点 → 开箱看词条 → 升2级
  P1: 附近有怪 → 刷到3级
  P2: 看周围: 要塞/教堂好打→打 | 不好打→野外Boss/下矿

发育阶段:
  P3: 野外Boss → 先3级再打(安全)
  P4: 看圈: 允许→主城二楼→6级→下水道 | 来得及+圈合适→楼顶(贪)
       圈不合适→固定野外Boss点
  P5: 顺路: 火车营地/癫火营地
  目标: 第一天摸到一个终端

成型阶段:
  P6: 第二天 → 野外红名 ×2 → 稳定15级
       穿插其他Boss/遗迹营地
  P7: 要塞/教堂 → 开局后基本不打

通用原则:
  · 词条: 战技/近战/满血攻击力=留 | 魔力/智力=丢
  · 商店: 有攻击词条→抓
  · 有对应终端→补对应属性攻击
  · 法师塔: 会开的解法→顺路做 | 不会的→跳过
```

## 已知限制 & 解决方案

| 限制 | AI 表现 | 解决方案 |
|------|---------|----------|
| 不会读词条 | 装备选择随机 | Phase 1.5 OCR 规则引擎 |
| 不认地图 | 不知道去哪 | 规划脑地图工具 + 规则 |
| 不会选路线 | 无法自主导航 | 规划脑路线优先级表 |
| 不会开法师塔 | 新解法站门口发呆 | 规划脑跳过 / VPT 学视频 |
| 不会平台跳跃 | 特殊地形卡住 | 固定路线多录 |
| 暴毙拔线 | 学到"残血=退出" | 预处理清洗裁段 |

## 训练原则

- 数据质量 > 数据数量
- 底座共用，决策层按游戏/模式切换
- 纯视觉 + 模拟输入，不碰游戏内存
- 训练和推理环境统一（分辨率/画质一致）
- 战斗脑(BC) + 规划脑(规则/LLM) 双脑分离
- 同一路线固定走，别换

## 硬件规划

| 阶段 | 用哪台 |
|------|------|
| 现在录制 | 台式机（2K 录制 → 720p 存盘）|
| 现在训练 | 台式机 4060M / 未来 5070 |
| 未来推理-单通 | 台式机 |
| 未来推理-队友 | 笔记本（跑客户端+AI）|
| 离线批量训练 | 笔记本后台慢慢跑 |
| 录制存储 | X盘（932G, 剩余300G → ~40局容量）|

## 参考资料

### 已下载论文（X:\dev\ai_books\）

- CSGO_BC_2021.pdf — CS:GO Behavioural Cloning, IEEE 最佳论文
- DarkSouls_Combat_2025.pdf — 纯像素训 AI 打魂类
- 强化学习导论_第二版_中文版.pdf
- 深度学习_花书_中文版.pdf

### 必看项目（GitHub）

| 项目 | 链接 | 看点 |
|------|------|------|
| **NVIDIA NitroGen** | [github.com/MineDojo/NitroGen](https://github.com/MineDojo/NitroGen) | 视觉→手柄，一模型打 1000+ 游戏，40K 小时视频训练 |
| **CS:GO Behavioural Cloning** | [github.com/TeaPearce/Counter-Strike_Behavioural_Cloning](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning) | IEEE 最佳论文，完整录制→训练→部署流水线，700G 数据集 |
| **Microsoft Visual Encoders** | [github.com/microsoft/imitation_learning_in_modern_video_games](https://github.com/microsoft/imitation_learning_in_modern_video_games) | 视觉编码器选型指南，MC + CS:GO 上的模仿学习 |
| **gamepy** | [github.com/jasonrobwebster/gamepy](https://github.com/jasonrobwebster/gamepy) | 极简录制工具（截图+按键→CSV），带 Keras 示例 |
| **OWL Control** | [github.com/Overworldai/owl-control](https://github.com/Overworldai/owl-control) | 支持键盘+鼠标+**手柄**录制 |
| **Unity ML-Agents** | [github.com/Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents) | 17K Star，PPO+模仿学习+自博弈，训练方法论最佳参考 |
