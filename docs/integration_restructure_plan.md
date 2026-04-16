# 主框架与 HoloOcean 仿真环境融合重构方案

本文档只描述目录重构和架构收敛方案，不要求立刻修改现有实现。目标是先明确边界，再按阶段迁移，避免直接移动目录后把当前工程打坏。

## 1. 重构目标

你当前的总体构想可以稳定落成 4 层：

- 上层 `LLM`：接收自然语言高层指令，做意图解析、任务分解、情境理解
- 中层 `Orchestration`：把结构化子任务编排成可执行序列，决定什么时候切技能、切场景、切传感器
- 下层 `RL Skills`：每个强化学习智能体只负责一种低层能力，如导航、避障、搜索、精细测绘
- 底层 `Simulation`：统一封装 HoloOcean 场景、传感器、动作映射、状态回传

重构后的框架要满足 3 个核心要求：

1. 新增仿真场景时，不需要反复改主入口和已有任务代码。
2. 新增 RL 子任务时，不需要改一堆 `if/else` 或手工模式索引。
3. HoloOcean 上游运行时与业务逻辑分离，后续升级版本时代价可控。

## 2. 当前仓库存在的主要问题

基于现有代码结构，主要问题有：

- `Main-Framework/main.py` 中直接写死了模型路径、模型类型、任务文本、模式切换流程。
- `Main-Framework/task/task_combine1.py` 直接实例化 `Navigation` 和 `Obstacles_Avoidance`，扩展新任务时必须改源码。
- `Main-Framework/llm/Interface.py` 用字符串任务名和 mode index 做映射，扩展性差。
- `Main-Framework/env/env_config.py` 把场景、传感器、奖励、日志、训练参数等全部堆在一个大配置里，职责不清。
- `Main-Framework/env/pierharbor_hovering.py` 把“基础环境能力”和“PierHarbor 场景细节”耦合在一起。
- `HoloOcean-2.3.0` 目录里既有上游运行时，也混有你们自己的旧实验环境思路，边界不明确。

这些问题会导致后期一旦增加：

- 新场景，如 `Dam`、`OpenWater`
- 新传感器，如 `ImagingSonar`、`RGBCamera`、`BST`
- 新技能，如 `target_tracking`、`search`、`mapping`

就需要同时修改多个文件，主框架会越来越脆弱。

## 3. 重构原则

本项目建议遵守以下原则：

### 3.1 上游运行时与业务代码分离

`HoloOcean-2.3.0/client` 和 `HoloOcean-2.3.0/engine` 本质上是底层运行时，不应直接混在业务目录中。它们更适合被放入一个 `third_party` 或 `vendor` 目录。

### 3.2 场景与任务分离

场景回答的是“世界怎么搭”，任务回答的是“智能体在这个世界里要优化什么”。这两者不能继续写在一个类或一个配置文件里。

### 3.3 技能与编排分离

RL 智能体应被看作多个低层技能执行器，每个技能单独训练、单独配置、单独评估。负责任务切换的逻辑应放在编排层，而不是放在任务环境类里硬编码。

### 3.4 配置驱动，而不是硬编码驱动

后期新增场景或技能时，应尽量通过“新建模块 + 注册配置”完成，而不是改主入口。

## 4. 建议的目标目录结构

建议最终将仓库整理成如下结构：

```text
611-Smart-Captain/
├── README.md
├── docs/
│   └── integration_restructure_plan.md
├── third_party/
│   └── holoocean/
│       ├── client/
│       ├── engine/
│       └── ...
├── src/
│   └── smart_captain/
│       ├── app/
│       │   └── main.py
│       ├── llm/
│       │   ├── interface.py
│       │   ├── decomposer.py
│       │   ├── world_model.py
│       │   └── prompts/
│       ├── orchestration/
│       │   ├── planner.py
│       │   ├── dispatcher.py
│       │   ├── task_graph.py
│       │   └── registry.py
│       ├── simulation/
│       │   ├── core/
│       │   │   ├── base_env.py
│       │   │   ├── action_mapper.py
│       │   │   ├── sensor_processor.py
│       │   │   └── scenario_manager.py
│       │   ├── scenarios/
│       │   │   ├── pier_harbor.py
│       │   │   ├── dam.py
│       │   │   └── open_water.py
│       │   ├── sensors/
│       │   │   ├── radar.py
│       │   │   ├── imaging_sonar.py
│       │   │   ├── rgb_camera.py
│       │   │   └── bst.py
│       │   └── registry.py
│       ├── skills/
│       │   ├── base.py
│       │   ├── navigation/
│       │   │   ├── env.py
│       │   │   ├── reward.py
│       │   │   ├── config.py
│       │   │   ├── policy.py
│       │   │   └── train.py
│       │   ├── obstacle_avoidance/
│       │   ├── target_tracking/
│       │   ├── search/
│       │   └── mapping/
│       ├── rl/
│       │   ├── agents.py
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   ├── model_store.py
│       │   └── algo_config.py
│       ├── configs/
│       │   ├── scenarios/
│       │   ├── sensors/
│       │   ├── tasks/
│       │   └── models/
│       └── utils/
│           ├── logging.py
│           ├── paths.py
│           └── types.py
├── models/
│   ├── llm/
│   └── rl/
├── outputs/
│   ├── logs/
│   ├── checkpoints/
│   └── eval/
└── scripts/
    ├── run_demo.py
    ├── train_skill.py
    ├── eval_skill.py
    └── manual_control.py
```

## 5. 两个现有主目录的定位建议

### 5.1 `Main-Framework`

这个目录已经具备“主框架雏形”，但其内部仍然是以实验迭代方式堆起来的。建议后续逐步把里面真正有长期价值的代码迁移到 `src/smart_captain/` 下，让它成为唯一主框架。

### 5.2 `HoloOcean-2.3.0`

这个目录不建议整体合并进主业务层。更合理的方式是把它拆成两部分看待：

- 上游运行时：`client`、`engine` 这类内容，归档到 `third_party/holoocean/`
- 你们自己的旧实验环境和脚本：只抽取有价值的思路和模块，迁移到 `simulation/`、`skills/`、`scripts/`

核心结论是：

- 不要把 `HoloOcean-2.3.0` 整个并到 `Main-Framework`
- 也不要继续让两个大目录并行长期发展

最终应该只有一个主框架目录，外加一个清晰的第三方运行时目录。

## 6. 当前文件到目标结构的迁移映射

下面给出基于现有仓库的建议迁移表。

### 6.1 `Main-Framework` 侧

| 当前路径 | 建议目标位置 | 说明 |
|---|---|---|
| `Main-Framework/main.py` | `src/smart_captain/app/main.py` | 只保留依赖装配和主流程启动 |
| `Main-Framework/llm/Interface.py` | `src/smart_captain/llm/interface.py` | 对外 LLM 接口层 |
| `Main-Framework/llm/Task_decomposition.py` | `src/smart_captain/llm/decomposer.py` | 只负责任务分解 |
| `Main-Framework/env/agents.py` | `src/smart_captain/rl/agents.py` | 模型加载、切换、预测 |
| `Main-Framework/env/trainer.py` | `src/smart_captain/rl/trainer.py` | 强化学习训练封装 |
| `Main-Framework/train/evaluator.py` | `src/smart_captain/rl/evaluator.py` | 统一评估逻辑 |
| `Main-Framework/env/env_config.py` | 拆入 `configs/scenarios/`、`configs/sensors/`、`configs/tasks/` | 大配置字典应拆分 |
| `Main-Framework/env/pierharbor_hovering.py` | `simulation/core/base_env.py` + `simulation/scenarios/pier_harbor.py` | 基类与场景分离 |
| `Main-Framework/task/task1.py` | `skills/navigation/env.py` | 导航技能 |
| `Main-Framework/task/task2.py` | `skills/obstacle_avoidance/env.py` | 避障技能 |
| `Main-Framework/task/task_combine.py` | `orchestration/dispatcher.py` | 多任务切换统一收口 |
| `Main-Framework/task/task_combine1.py` | `orchestration/dispatcher.py` 或 `planner.py` | 不再做硬编码双任务切换 |
| `Main-Framework/control.py` | `scripts/manual_control.py` | 手控脚本归入工具层 |
| `Main-Framework/MPC/mpc_controler.py` | `skills/path_tracking/` 或 `controllers/mpc/` | 后续路径跟踪能力 |
| `Main-Framework/logs*` | `outputs/logs/` | 输出统一归档 |
| `Main-Framework/stable_baselines3` | 暂时保留或转 vendor 目录 | 不建议继续扩散依赖边界 |

### 6.2 `HoloOcean-2.3.0` 侧

| 当前路径 | 建议目标位置 | 说明 |
|---|---|---|
| `HoloOcean-2.3.0/client` | `third_party/holoocean/client` | 上游运行时 |
| `HoloOcean-2.3.0/engine` | `third_party/holoocean/engine` | 上游运行时 |
| `HoloOcean-2.3.0/BaseEnv/command与推进器映射.py` | `simulation/core/action_mapper.py` | 抽成动作映射工具 |
| `HoloOcean-2.3.0/BaseEnv` 中各类任务思路 | 拆分迁移到 `skills/`、`simulation/scenarios/`、`scripts/` | 不建议保留旧目录形态 |
| 传感器测试脚本 | `scripts/sensor_tests/` | 作为实验工具保留 |

## 7. 推荐的架构边界

### 7.1 `simulation/` 负责什么

`simulation/` 只负责仿真底座，职责包括：

- 加载和关闭 HoloOcean 环境
- 注册场景、传感器、智能体初始配置
- 处理原始传感器回传
- 将高层动作映射为 HoloOcean 底层控制指令
- 向上提供统一的 `reset()`、`step()`、`close()`

`simulation/` 不应负责：

- LLM 任务分解
- 模型路径选择
- 技能切换
- 高层任务规划

### 7.2 `skills/` 负责什么

`skills/` 应该表示一组低层可执行能力，每个技能目录单独管理：

- 观测构造
- 动作空间定义
- 奖励函数
- 终止条件
- 训练与评估入口
- 对应模型权重

建议至少预留以下技能目录：

- `navigation`
- `obstacle_avoidance`
- `target_tracking`
- `search`
- `mapping`
- `path_tracking`

### 7.3 `orchestration/` 负责什么

这一层是主框架中最关键的中间层，负责：

- 接收 LLM 输出的结构化子任务
- 决定当前应调用哪个技能
- 管理技能切换时的上下文传递
- 协调场景参数、传感器模式、任务状态
- 判断某个子任务何时结束并进入下一个

当前 `task_combine1.py` 的作用其实已经接近这个层，但实现方式还是硬编码的，后期应被注册表机制替换。

### 7.4 `llm/` 负责什么

建议把 LLM 层拆成 3 个角色：

- `decomposer.py`：把高层自然语言拆成任务序列
- `world_model.py`：融合海洋环境、潜航器能力、任务协议等知识
- `interface.py`：对外暴露统一接口，输出结构化任务图

## 8. LLM 输出形式的建议

当前实现更像是：

- 把一句高层指令分解成若干字符串
- 再映射成 mode index

这种方式不适合长期扩展。建议后续升级成结构化输出，而不是仅仅输出索引列表。

推荐输出形式：

```python
[
    {
        "skill": "navigation",
        "goal": {"x": 30, "y": 12, "z": -8},
        "constraints": {"avoid_obstacles": True},
        "sensor_mode": "rangefinder",
        "success_condition": "reach_waypoint"
    },
    {
        "skill": "search",
        "area": "sector_A3",
        "pattern": "lawnmower",
        "sensor_mode": "imaging_sonar",
        "success_condition": "target_candidate_detected"
    },
    {
        "skill": "mapping",
        "target": "possible_wreck",
        "sensor_mode": "high_res_sonar",
        "success_condition": "mapping_completed"
    }
]
```

这样做的好处是：

- 编排层不用依赖脆弱的 mode index
- 新增技能时，LLM 侧只需要产出新的 `skill` 标识和参数
- 各技能可以接收自身真正需要的参数，而不是共享一套模糊输入

## 9. 场景扩展机制建议

为了方便后期增加仿真场景，建议引入 `SCENARIO_REGISTRY`。

示例：

```python
SCENARIO_REGISTRY = {
    "pier_harbor": "smart_captain.simulation.scenarios.pier_harbor:PierHarborScenario",
    "dam": "smart_captain.simulation.scenarios.dam:DamScenario",
    "open_water": "smart_captain.simulation.scenarios.open_water:OpenWaterScenario",
}
```

新增场景时，理想步骤应只有：

1. 新增一个场景模块
2. 注册到 `SCENARIO_REGISTRY`
3. 在技能配置中声明默认场景

而不是去修改主入口、环境主类、任务切换器多个位置。

## 10. RL 子任务扩展机制建议

为了方便后期增加 RL 可执行子任务，建议引入 `SKILL_REGISTRY`。

示例：

```python
SKILL_REGISTRY = {
    "navigation": {
        "env_cls": "smart_captain.skills.navigation.env:NavigationEnv",
        "model_path": "models/rl/navigation/latest.zip",
        "scenario": "pier_harbor",
    },
    "obstacle_avoidance": {
        "env_cls": "smart_captain.skills.obstacle_avoidance.env:ObstacleAvoidanceEnv",
        "model_path": "models/rl/obstacle_avoidance/latest.zip",
        "scenario": "open_water",
    },
    "search": {
        "env_cls": "smart_captain.skills.search.env:SearchEnv",
        "model_path": "models/rl/search/latest.zip",
        "scenario": "dam",
    },
}
```

以后新增一个子任务，理想情况下只需要：

1. 新建一个 `skills/<skill_name>/`
2. 编写它的环境、奖励、训练脚本和模型
3. 在 `SKILL_REGISTRY` 中注册

这样主入口和任务编排层都不需要频繁重写。

## 11. 针对当前代码，最应该优先解决的 5 个问题

如果后续真的开始动代码，建议优先处理下面 5 件事：

### 11.1 去掉 `main.py` 中的硬编码

当前 `main.py` 中直接写死了：

- `model_paths`
- `model_types`
- `main_task`

这些都应改为配置驱动。

### 11.2 去掉 `task_combine1.py` 中对具体任务类的硬编码

当前直接写死 `Navigation` 和 `Obstacles_Avoidance`，这意味着每增加一个任务都要手改此文件。后续应改为从注册表动态实例化。

### 11.3 把 `pierharbor_hovering.py` 拆为基类和场景层

当前这个文件混合了：

- HoloOcean 环境初始化
- 观测与动作定义
- 目标点生成
- 雷达处理
- 奖励统计

后续应把“基础交互能力”和“具体场景逻辑”拆开。

### 11.4 把 `env_config.py` 拆成多个配置文件

建议至少分为：

- 场景配置
- 传感器配置
- 技能配置
- 奖励配置
- 训练配置

### 11.5 把 LLM 输出升级成结构化任务图

当前输出 mode index 不足以支撑“情境理解 + 动态任务切换 + 参数传递”的目标构想，后续必须升级。

## 12. 推荐的迁移顺序

后续真的开始实施时，建议按下面顺序做，风险最小。

### 阶段 1：补足框架骨架，但不改现有行为

- 新建 `docs/`、`src/`、`third_party/`、`models/`、`outputs/`、`scripts/`
- 先不删除旧目录
- 先建立新结构占位

### 阶段 2：抽底层仿真适配层

- 从 `pierharbor_hovering.py` 中抽出 `base_env.py`
- 把动作映射和传感器处理抽为独立模块
- 场景定义单独放到 `simulation/scenarios/`

### 阶段 3：抽 RL 技能层

- 把 `task1.py` 迁移为 `skills/navigation/env.py`
- 把 `task2.py` 迁移为 `skills/obstacle_avoidance/env.py`
- 后续任务按同一模板接入

### 阶段 4：抽编排层

- 用 `dispatcher.py` 接管技能切换
- 用注册表替代 `task_combine1.py` 的硬编码实现

### 阶段 5：抽 LLM 结构化接口

- 把任务分解、知识增强、参数生成、结构化输出拆开
- 不再把 mode index 当成主接口

### 阶段 6：清理旧目录

- 确认新入口可运行后，再逐步移除或归档 `Main-Framework` 的旧结构
- `HoloOcean-2.3.0/BaseEnv` 作为参考代码归档，而不是长期主代码

## 13. 一个更贴合项目构想的最终运行链路

建议最终的主流程是：

```text
人类自然语言指令
-> LLM 任务解析
-> 结构化任务图
-> Orchestration 编排器
-> 激活当前 RL Skill
-> 调用 Simulation 执行
-> 传感器回传
-> 更新世界状态
-> 判断是否进入下一子任务
```

对应的伪代码可以是：

```python
command = "导航到目标区域，避障搜索可疑沉船，发现后进行精细测绘"

task_graph = llm_interface.parse(command, world_state)

for subtask in task_graph:
    skill = dispatcher.activate(subtask["skill"], subtask)
    result = skill.run_until_done(world_state)
    world_state = result.updated_world_state
```

这比“LLM 一次性给出 mode 序列，然后 RL 从头跑到尾”的方式更贴合你的原始构想，也更适合处理动态感知和中途重规划。

## 14. 结论

如果只给一句落地建议，那就是：

不要先直接移动整个目录，而是先把项目明确拆成 `llm / orchestration / skills / simulation / rl / third_party` 六层，再按迁移顺序逐步收敛。

最关键的组织原则有 3 条：

1. `HoloOcean` 作为底层运行时保留在独立第三方目录，不直接和业务层混写。
2. `Main-Framework` 中真正有用的代码最终应收敛为唯一主框架，并按语义分层。
3. 未来新增场景和新增 RL 子任务，都应通过“新增模块 + 注册表”完成，而不是继续改主入口和硬编码切换逻辑。

只要按这个方向推进，后面无论是加新仿真场景，还是加新的低层可执行技能，接入成本都会明显下降。
