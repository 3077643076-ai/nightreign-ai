# 黑环AI — 强化学习打黑夜君临

训练一个能当队友/单通的通用游戏 AI，从《艾尔登法环 黑夜君临》起步。

**当前角色**：追踪者（物理战技弹反处决流）

## 核心理念：双脑分离

黑环决策天然分为两层，用不同方案各取所长：

```
┌─────────────────────┐    ┌─────────────────────┐
│  战斗脑 (帧级)       │    │  规划脑 (秒/分钟级)    │
│  CNN 行为克隆        │    │  OCR + 规则引擎 + LLM │
│  画面 → 翻滚/弹反/攻击 │    │  文字 → 选装备/路线    │
│  BC 能学好 ✅        │    │  BC 学不了, 用规则 ✅  │
└─────────────────────┘    └─────────────────────┘
```

## 技术栈

| 用途 | 工具 |
|------|------|
| 截图 | DXCam（2K → 720p / 15fps） |
| 手柄录制 | `inputs` 库 (XInput) |
| 虚拟手柄 | vgamepad + ViGEmBus |
| 深度学习 | PyTorch + CUDA |
| 目标检测 | YOLOv8 |
| RL 框架 | stable-baselines3 |
| OCR | Tesseract（规划脑） |

## 快速开始

### 环境

- Windows 10/11 + NVIDIA 显卡（训练用 4060M / 5070）
- Python 3.10+
- [ViGEmBus](https://github.com/nefarius/ViGEmBus) 驱动（虚拟手柄）

### 安装

```bash
git clone git@github.com:3077643076-ai/nightreign-ai.git
cd nightreign-ai
pip install -r requirements.txt
```

### 录制数据

```bash
# 双击 run.bat（自动提权+检查依赖）
# 或命令行：
python record.py

# 热键：F8 开始录制 / F9 停止 / Ctrl+C 退出
```

录制数据保存在 `data/` 目录，格式为 720p JPEG 截图 + `inputs.jsonl`（手柄按键+摇杆）。

### 训练

```bash
# 基础行为克隆（CNN：画面→手柄按键+摇杆）
python run_train.py

# 评估模型
python eval_bc.py
```

### 运行 AI

```bash
# 启动 AI 推理（需要先训练好模型）
python run_ai.bat
```

## 项目结构

```
game-ai-agent/
├── record.py              # 录制：截图+手柄输入
├── recorder/              # 录制工具集
├── preprocess/            # 数据预处理、降质、清洗
├── models/                # 模型定义（CNN/IDM/LSTM）
├── train/                 # 训练脚本（BC + 分类器）
├── inference/             # 推理部署
├── planner.py             # 规划脑：规则引擎
├── game_state.py          # 游戏状态解析
├── combat_detector.py     # 战斗检测器
├── enemy_detector.py      # 敌人检测（YOLOv8）
├── healthbar_detector.py  # 血条HUD解析
├── keyboard_controller.py # 键盘控制
├── perception/            # 感知模块（OCR等）
├── data/                  # 录制数据（gitignore）
├── checkpoints/           # 模型权重（gitignore）
├── tools/                 # 辅助工具
└── ROADMAP.md             # 完整路线图
```

## 进度

| 阶段 | 状态 |
|------|------|
| Phase 0 — 录制工具 | ✅ 完成（60K+ 帧已录，14.8fps） |
| Phase 1 — 战斗底座（BC训练） | 🔄 进行中（2局 / 目标15+局） |
| Phase 1.5 — OCR装备管理 | ⏳ 规划中 |
| Phase 2 — VPT视频消化 | ⏳ 规划中 |
| Phase 3 — RL自我博弈 | ⏳ 规划中 |
| Phase 4 — 队友模式 | ⏳ 规划中 |
| Phase 5 — 迁移其他游戏 | ⏳ 怪猎/死亡细胞/MC |

## 路线规则（追踪者·物理流）

```
落地 → 最近据点 → 开箱看词条 → 升2级
附近有怪 → 刷到3级
野外Boss → 先3级再打
看圈 → 主城二楼 → 6级 → 下水道
第二天 → 野外红名 ×2 → 稳定15级
```

## 训练原则

- 数据质量 > 数据数量
- 纯视觉 + 模拟输入，不碰游戏内存
- 战斗脑(BC) + 规划脑(规则/LLM) 双脑分离
- 同一路线固定走，别换

## 参考

- [CS:GO Behavioural Cloning](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning) — IEEE 最佳论文，700G 数据集
- [NVIDIA NitroGen](https://github.com/MineDojo/NitroGen) — 视觉→手柄，一模型打 1000+ 游戏
- [gamepy](https://github.com/jasonrobwebster/gamepy) — 极简录制工具（截图+按键）
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) — PPO+模仿学习+自博弈

## 许可

MIT
