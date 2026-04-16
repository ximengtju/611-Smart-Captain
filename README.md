# 🚢 611-Smart-Captain

> LLM + RL 分层智能控制框架，面向水下无人潜航器（AUV）自主任务执行

611-Smart-Captain 是一个分层协同控制系统，支持通过自然语言下达高层任务指令，由系统自动完成任务理解、分解、编排与执行，并在 HoloOcean 仿真环境中进行训练与验证。

核心思路是 **"LLM 当船长🧠，RL 当舵手🎮"**：

- **LLM（大语言模型）** 负责"想"——理解自然语言指令，将任务拆解为结构化子任务序列
- **RL（强化学习）** 负责"做"——将子目标转化为推进器推力、舵角等底层控制指令
- **Orchestration（编排层）** 负责"调度"——在技能间切换、管理传感器模式、推进任务状态

## 🏗️ 系统架构

系统分为四层：

```
自然语言指令
    ↓
🧠 LLM 层 ─── 意图解析、任务分解、情境理解
    ↓
🧭 编排层 ─── 技能切换、传感器切换、执行状态管理
    ↓
🎮 RL 层 ──── 推进器控制、舵角控制、路径执行
    ↓
🌊 仿真层 ─── HoloOcean 水下物理仿真、传感器模拟
    ↓
📡 环境反馈 → 回传至编排层与 LLM 层
```

**🧠 LLM 任务理解层** — 接收自然语言指令（如"前往目标海域，途中避障，搜索可疑目标，切换声呐精细测绘"），完成意图解析、任务分解（导航 → 避障 → 搜索 → 传感器切换 → 测绘）和情境理解（如将声呐图像中的模糊形状关联到"可能的沉船"）。

**🧭 Orchestration 编排层** — 将结构化子任务编排为可执行流程，在导航/避障/搜索/跟踪/测绘等技能之间切换，管理场景与传感器模式，跟踪执行状态与依赖关系。

**🎮 RL 执行层** — 强化学习智能体将子目标转化为底层控制指令，处理复杂非线性水下动力学，优化能耗与任务完成时间，在噪声与不确定环境下保持稳定执行。

**🌊 仿真层** — 基于 [HoloOcean](https://holoocean.readthedocs.io/) 提供水下场景仿真、AUV 运行时、传感器观测与训练评估环境。

> **技能（Skill）**：指一个可被编排层调度、由 RL 层执行的任务模块，例如 `navigation`、`obstacle_avoidance`、`search`、`mapping`。通常对应一个技能环境、一个策略入口，以及该任务相关的观测/奖励逻辑。

## 🧭 当前编排层

当前编排层采用的是一套**线性任务图 + 确定性调度器**架构。

- **输入**：LLM 层输出的结构化子任务列表，每个子任务包含 `skill`、`scenario`、`sensor_mode`、`constraints` 等字段
- **输出**：当前应激活的技能、场景、传感器模式，以及对应环境类和场景类
- **执行时机**：
  - 规划阶段：把结构化任务转成 `TaskGraph`
  - 推进阶段：根据任务结果把当前子任务从 `pending` 推进到 `active / succeeded / failed`
- **当前已实现的能力**：
  - 线性任务图构建
  - 子任务状态管理
  - 技能/场景/环境解析
  - 当前激活任务的选择
  - 执行上下文维护

### `orchestration/` 各文件功能

- `task_graph.py`
  定义 `Subtask`、`TaskGraph`、`ExecutionContext`，是编排层的核心数据结构
- `planner.py`
  把 LLM 产出的结构化任务 payload 转成可执行的 `TaskGraph`
- `dispatcher.py`
  负责激活当前子任务、更新状态、推进到下一个子任务
- `registry.py`
  负责把技能名、场景名、传感器名解析到具体实现类
- `multi_env.py`
  提供共享 AUV 实例下的多环境适配逻辑，支撑多技能切换

### 还需要继续完善的部分

- 目前只支持**线性任务链**，还不支持分支、并行和 DAG 依赖
- 还没有实现真正的**重规划**和失败恢复
- 还没有形成“观察 -> 判断 -> 切换 -> 再观察”的完整 agent 闭环
- 传感器切换和场景切换目前主要来自任务分解结果，还不是基于实时反馈的动态决策

## 📁 项目结构

### `src/smart_captain/` — 核心代码

```text
src/smart_captain/
├── app/                           应用入口
│   ├── mission_runner.py          统一任务入口，串联规划、编排与执行
│   ├── main.py                    轻量规划演示入口
│   ├── compat_runtime.py          兼容运行时
│   ├── native_runtime.py          原生运行时
│   └── shared_auv_runtime.py      共享 AUV 运行时
├── llm/                           任务理解
│   ├── interface.py               LLM 统一接口
│   ├── decomposer.py              任务分解器
│   └── world_model.py             世界模型与能力画像
├── orchestration/                 任务编排
│   ├── task_graph.py              子任务、任务图、执行上下文
│   ├── planner.py                 结构化任务 -> 任务图
│   ├── dispatcher.py              执行推进与技能切换
│   ├── registry.py                技能/场景/传感器解析
│   └── multi_env.py               共享 HoloOcean 实例的多环境适配器
├── rl/                            强化学习
│   ├── model_store.py             技能 -> 模型权重注册表
│   ├── agents.py                  多技能策略运行时
│   ├── algo_config.py             算法超参数配置
│   ├── trainer.py                 训练封装
│   └── evaluator.py               评估封装
├── simulation/                    仿真环境
│   ├── defaults.py                默认环境配置
│   ├── compat.py                  迁移兼容层
│   ├── registry.py                场景与传感器注册表
│   ├── core/
│   │   ├── base_env.py            仿真基础类与奖励数学工具
│   │   ├── action_mapper.py       RL 动作 -> HoloOcean 控制指令
│   │   ├── sensor_processor.py    传感器数据预处理
│   │   └── scenario_manager.py    场景管理器
│   ├── scenarios/
│   │   ├── pier_harbor.py         港口码头场景
│   │   ├── dam.py                 大坝场景
│   │   └── open_water.py          开阔水域场景
│   └── sensors/
│       ├── radar.py               雷达传感器适配器
│       ├── imaging_sonar.py       成像声呐适配器
│       ├── rgb_camera.py          RGB 摄像头适配器
│       └── bst.py                 BST 传感器适配器
├── skills/                        可执行技能
│   ├── base.py                    SkillSpec 与 SkillAdapter
│   ├── registry.py                技能注册表
│   ├── navigation/
│   │   ├── config.py              导航技能配置
│   │   ├── env.py                 导航技能环境
│   │   ├── policy.py              导航策略封装
│   │   ├── reward.py              导航奖励函数
│   │   └── train.py               导航技能训练入口
│   ├── obstacle_avoidance/
│   │   ├── config.py              避障技能配置
│   │   ├── env.py                 避障技能环境
│   │   ├── policy.py              避障策略封装
│   │   ├── reward.py              避障奖励函数
│   │   └── train.py               避障技能训练入口
│   ├── search/
│   │   ├── env.py                 搜索技能环境，占位
│   │   └── policy.py              搜索策略封装，占位
│   ├── target_tracking/
│   │   ├── env.py                 目标跟踪技能环境，占位
│   │   └── policy.py              目标跟踪策略封装，占位
│   └── mapping/
│       ├── env.py                 测绘技能环境，占位
│       └── policy.py              测绘策略封装，占位
└── utils/                         工具模块
    ├── paths.py                   路径工具
    ├── types.py                   类型定义
    ├── logging.py                 日志工具
    └── compat_testbeds.py         适配器测试用模拟环境
```

### `src/stable_baselines3/` — 定制版 Stable Baselines3

项目使用的 RL 训练库（基于官方版本定制修改），支持 SAC、PPO、A2C、DDPG、DQN、TD3 等算法。

### `src/holoocean/` — HoloOcean Python Client

[HoloOcean](https://holoocean.readthedocs.io/) 水下仿真引擎的 Python 客户端，负责与仿真引擎通信、管理传感器和智能体。

### 其他顶层目录

| 目录 | 说明 |
|---|---|
| `models/rl/` | 已训练的 RL 模型权重（navigation/SAC、obstacle_avoidance/SAC） |
| `docs/` | 设计文档 |
| `third_party/holoocean/engine/` | HoloOcean 仿真引擎 C++ 源码 |

### `scripts/` — 📜 运行脚本

```text
scripts/
├── run_mission_runner.py          主入口，支持 plan-only / legacy-preview / execute
├── run_native_preview.py          预览原生运行时规划结果
├── run_legacy_compat.py           验证兼容运行链
├── run_demo.py                    演示脚本
├── train_skill.py                 技能训练脚本
├── eval_skill.py                  技能评估脚本
└── manual_control.py              手动控制 AUV
```

## 🚀 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

验证环境：

```bash
python -c "import torch, gymnasium, holoocean, stable_baselines3"
```

如果希望任务分解优先调用真实 LLM API，而不是本地关键词规则，可以配置 DeepSeek：

```bash
export SMART_CAPTAIN_LLM_API_URL="https://api.deepseek.com/chat/completions"
export SMART_CAPTAIN_LLM_API_KEY="你的 DeepSeek API Key"
export SMART_CAPTAIN_LLM_MODEL="deepseek-chat"
```

未配置这些环境变量时，框架会继续使用当前内置的规则分解器。

### 运行示例

```bash
# 预览任务规划结果（不启动仿真）
PYTHONPATH=src python scripts/run_mission_runner.py --mode plan-only

# 预览运行链绑定
PYTHONPATH=src python scripts/run_mission_runner.py --mode legacy-preview

# 执行完整任务链（需要仿真环境和模型权重就绪）
PYTHONPATH=src python scripts/run_mission_runner.py --mode execute

# 自定义任务指令
PYTHONPATH=src python scripts/run_mission_runner.py \
  --mode plan-only \
  --command "请控制水下机器人前往目标区域，途中避障，然后开始搜索可疑目标"
```

`plan-only` 输出中的 `llm_backend` 字段表示本次任务分解使用的是哪条链路：

- `heuristic`：使用本地关键词规则分解
- `api`：使用已配置的真实 LLM API，例如 DeepSeek

### 训练与评估

```bash
PYTHONPATH=src python scripts/train_skill.py
PYTHONPATH=src python scripts/eval_skill.py
```

## 🧩 开发指南

### 新增技能

1. 在 `src/smart_captain/skills/` 下新建技能目录：

```
src/smart_captain/skills/<skill_name>/
  __init__.py
  config.py       # 技能配置
  env.py          # 技能环境
  policy.py       # 策略入口
  reward.py       # 奖励函数
  train.py        # 训练脚本
```

2. 在 `skills/registry.py` 中注册技能
3. 在 `rl/model_store.py` 中注册默认模型

### 新增任务类型

修改 `llm/decomposer.py` 中的关键词映射表，并在 `orchestration/planner.py` 中建立技能映射。

### 接入新模型

将模型权重放置到 `models/rl/<skill_name>/<algorithm>/<run_name>/`，并在 `rl/model_store.py` 中注册。

## 🗺️ Roadmap

- [ ] 更多技能：路径跟踪、精细测绘、目标确认、返航、协同搜索
- [ ] 高级编排：条件分支、失败重规划、任务恢复、动态重排序
- [ ] 持续情境理解：LLM 结合传感器反馈与世界模型进行实时解释
- [ ] 统一训练评估接口
- [ ] 更多场景与多模态传感器配置
- [ ] 完整 CLI 与实验管理工具

## 📄 License

MIT
