# MPC 接入 `target_tracking` 任务交接说明

更新时间：2026-05-25

## 1. 任务目标

本次工作的目标是将原 HoloOcean 工程中的 MPC 轨迹跟踪能力接入 Smart Captain，使任务规划器能够把 `target_tracking` 任务交给 MPC 控制器执行，并能够在真实 HoloOcean runtime 中运行。

当前确定的设计是：

- MPC 不单独建立 `controller` 目录，而是归属于 `skills/target_tracking/`。
- `target_tracking` 使用 MPC policy，不依赖训练好的 RL 模型。
- 原有 `navigation`、`obstacle_avoidance` 继续使用 RL 模型。
- 运行时需要同时支持纯 MPC 任务和 RL/MPC 混合任务。

## 2. 原始代码来源

MPC 接入所参考的旧工程文件位于 `HoloOcean-2.0.1`：

| 原文件 | Smart Captain 中的目标位置 | 用途 |
| --- | --- | --- |
| `MPC/mpc_controller.py` | `src/smart_captain/skills/target_tracking/mpc_controller.py` | MPC 优化控制器 |
| `MPC/mpc_predictor.py` | `src/smart_captain/skills/target_tracking/mpc_predictor.py` | AUV 运动预测模型 |
| `task/task3.py` | `src/smart_captain/skills/target_tracking/env.py` | 参考轨迹与观测构造逻辑 |
| `env/env_config.py` | `src/smart_captain/skills/target_tracking/config.py` | MPC 参数来源之一 |

讨论过程中曾考虑创建 `path_tracking` 技能，后续要求改为将 MPC 放入已有的 `target_tracking` 技能，因此最终应以 `target_tracking` 路径为准。

## 3. 当前已接入的文件

### 3.1 `skills/target_tracking` 下的实现

当前目录：

```text
src/smart_captain/skills/target_tracking/
    __init__.py
    config.py
    env.py
    evaluator.py
    mpc_controller.py
    mpc_predictor.py
    policy.py
```

各文件职责如下：

| 文件 | 当前作用 |
| --- | --- |
| `config.py` | 定义 `MPC_CONFIG` 和 `TARGET_TRACKING_SPEC` |
| `env.py` | 定义 `TargetTrackingEnv`，生成参考轨迹并拼接 MPC 观测 |
| `policy.py` | 定义 `TargetTrackingPolicy`，将 runtime 的 `predict()` 调用转发给 MPC |
| `evaluator.py` | 根据 `done`、`goal_reached`、`collision`、`truncated` 返回任务进度 |
| `mpc_controller.py` | 基于 CasADi/IPOPT 求解 MPC 控制量 |
| `mpc_predictor.py` | AUV 动力学预测模型，目前不直接承担 runtime policy 输出 |
| `__init__.py` | 导出该技能的 spec、环境和 policy |

### 3.2 关键参数

当前 `MPC_CONFIG` 中的重要参数：

```python
"horizon": 50
"dot_t": 0.005
"obs_space": 12
"command_space": 4
```

因此 `TARGET_TRACKING_SPEC` 中的观测维度为：

```python
12 + (50 + 1) * 12 = 624
```

含义为：

- 前 `12` 维：当前 AUV 状态。
- 后 `51 * 12` 维：MPC 预测范围内的参考状态轨迹。
- 输出动作为 `4` 维控制命令。

## 4. 已完成的代码修改

### 4.1 注册 `target_tracking` 技能

文件：`src/smart_captain/skills/registry.py`

当前已经导入并注册：

```python
from smart_captain.skills.target_tracking.config import TARGET_TRACKING_SPEC
```

```python
TARGET_TRACKING_SPEC.name: TARGET_TRACKING_SPEC
```

这样技能注册表使用真实的 MPC 技能规格，而不是原先的占位定义。

### 4.2 MPC policy 接口适配

文件：`src/smart_captain/skills/target_tracking/policy.py`

`TargetTrackingPolicy` 实例化 `AUV_MPC`，并提供与项目其他 policy 相同的接口：

```python
def predict(self, observation, state=None, deterministic=True):
    action = self.controller.predict(observation)
    return action, state
```

这使 runtime 不需要知道该策略内部是 RL 模型还是 MPC 求解器。

### 4.3 MPC 控制器配置传递

文件：`src/smart_captain/skills/target_tracking/mpc_controller.py`

构造函数当前使用传入配置：

```python
def __init__(self, config=MPC_CONFIG):
    self.config = config
```

这样外部传入的 MPC 参数会生效，便于后续调整 horizon、代价矩阵和控制边界。

### 4.4 TargetTracking 环境和 observation handoff

文件：`src/smart_captain/skills/target_tracking/env.py`

已完成的核心内容：

- `TargetTrackingEnv.spec = TARGET_TRACKING_SPEC`。
- 构造函数参数顺序与 runtime 创建方式兼容：

```python
def __init__(
    self,
    env_config: dict = BASE_CONFIG,
    auv=None,
    train_mode=True,
    mpc_config: dict = MPC_CONFIG,
):
```

- 增加 `build_tracking_observation()`，将当前状态和参考轨迹拼成 MPC 所需观测。
- `reset()` 和 `step()` 返回适用于 MPC 的观测。

当前 `reset()` 默认将路径缓存初始化为圆形轨迹，因此直接启动 `target_tracking` 时看到 AUV 沿圆环跟踪属于现有代码行为，不是 runtime 自动生成的错误路径。

### 4.5 mode 映射与 HoloOcean 环境绑定

文件：`src/smart_captain/app/bridge.py`

当前映射为：

```python
"navigation": 0
"obstacle_avoidance": 1
"target_tracking": 2
```

并在 `runtime_layout.task_bindings` 中新增了 mode `2` 对应的环境绑定：

```python
LegacyTaskBinding(
    mode_index=2,
    skill_name="target_tracking",
    env_import_path="smart_captain.skills.target_tracking.env:TargetTrackingEnv",
)
```

### 4.6 runtime 同时支持 MPC 与 RL

文件：`src/smart_captain/app/compat_runtime.py`

新增 `MixedSkillAgents`：

- mode `0`、`1` 时委托给 `LegacyCompatibleAgents` 的 RL 模型。
- mode `2` 时调用 `TargetTrackingPolicy` 的 MPC 控制器。
- 当 `rl_agents=None` 且错误切到 RL mode 时抛出清晰异常。

纯 MPC 任务的处理逻辑已加入两个 runtime 创建函数：

```python
if plan.mode_sequence and all(mode == 2 for mode in plan.mode_sequence):
    agents = MixedSkillAgents(
        rl_agents=None,
        extra_policies={2: TargetTrackingPolicy()},
    )
```

作用是：仅执行 `target_tracking` 时不加载 navigation 和 obstacle avoidance 的 RL 模型，避免无关模型依赖影响 MPC 测试。

此外，在技能切换后的 observation handoff 中，若目标技能为 `target_tracking`，当前 runtime 会使用：

```python
active_env.build_tracking_observation()
```

以确保传入 MPC 的观测维度和含义正确。

### 4.7 evaluator 注册

文件：

- `src/smart_captain/skills/target_tracking/evaluator.py`
- `src/smart_captain/orchestration/evaluator_registry.py`

`TargetTrackingEvaluator` 已被注册到 evaluator registry。目前其判断逻辑较基础：

- 碰撞或截断：失败。
- 到达目标或环境结束：成功。
- 其他情况：运行中。

后续如果要评估“轨迹跟踪质量”，还需要增加横向误差、姿态误差、速度误差等指标。

## 5. 当前运行行为

### 5.1 默认路径为何是圆环

`TargetTrackingEnv.reset()` 中直接调用了 `_generate_circular_path()`，并将其写入路径缓存。因此只运行 `target_tracking` 时，MPC 默认跟踪的是以目标点为圆心、以起点距离为半径的圆形路径。

如果后续要跟踪 planner 生成的 waypoints，需要将 waypoints 传入 `generate_reference_trajectory(..., path_steps=...)` 或建立对应的任务参数传递链。目前这部分尚未接入。

### 5.2 最大运行步数

项目默认环境配置位于：

```text
src/smart_captain/simulation/defaults.py
```

当前配置：

```python
"max_timesteps": 20000
```

需要区分：

- `max_timesteps`：一个真实 runtime episode 最多运行多少步。
- `MPC_CONFIG["horizon"]`：每一次 MPC 求解向未来预测多少步，目前为 `50`。

## 6. 已做的验证

在本轮接入过程中已做过以下检查：

| 验证项 | 结果 |
| --- | --- |
| `target_tracking` evaluator 导入 | 通过 |
| bridge 将轨迹跟踪任务映射到 mode `2` | 通过 |
| 纯 MPC plan 可跳过 RL 模型加载 | 通过轻量检查 |
| `MixedSkillAgents(rl_agents=None)` 在 mode `2` 调用 MPC | 通过轻量检查 |
| `TargetTrackingPolicy()` 实例化 | 通过，当前控制器参数为 `N=50`、`nx=12`、`nu=4` |
| navigation SAC 模型反序列化加载 | 更新 NumPy 后可以加载 |
| 真实 HoloOcean runtime | 曾进入 MPC/HoloOcean 执行流程；长时间运行测试仍需做有界验证 |

## 7. Python 环境变更

为处理 RL 模型加载时出现的 NumPy 反序列化错误，实际使用的 Python 环境：

```text
D:\Anaconda\python.exe
```

已更新以下包：

| 包 | 当前验证版本 |
| --- | --- |
| `numpy` | `2.4.6` |
| `scipy` | `1.17.1` |
| `scikit-image` | `0.26.0` |

更新后的验证结论：

- `numpy._core.numeric` 可以导入。
- 原先由 NumPy 模块路径导致的 SB3 模型反序列化错误已消失。
- MPC policy 可以导入并实例化。

需要注意：该环境是共享 Anaconda 环境，NumPy 2 可能与部分旧二进制包不兼容。此前测试输出中出现过 `pyarrow`、`numexpr`、`bottleneck` 的 ABI 兼容提示，以及旧版 `gym` 对 NumPy 2 的警告。纯 MPC 导入检查目前通过，但混合 RL/runtime 仍应继续验证。

## 8. 当前尚未完成或需要确认的问题

### 8.1 真实 runtime 的有限步数测试

目前仍需要稳定完成一次带明确步数上限的真实 HoloOcean `target_tracking` 测试，确认：

- HoloOcean 实际启动成功。
- MPC 能连续输出控制量。
- observation 维度始终为 `624`。
- 运行终止方式明确，不会因为 `max_timesteps=20000` 而长时间占用进程。

### 8.2 planner waypoints 尚未接给 MPC

当前 MPC 默认生成圆形参考轨迹，并未使用 planner 的 waypoints。后续需要定义：

- waypoints 放在任务图或 runtime world state 的哪个字段中。
- skill 切换时如何传入 `TargetTrackingEnv`。
- 没有 waypoints 时是否仍使用圆形测试路径。

### 8.3 evaluator 能力较简略

当前 evaluator 只判断终止状态，尚不能量化 MPC 的跟踪效果。建议至少添加：

- 当前点到参考轨迹的距离误差。
- 平均/最大跟踪误差。
- 控制量是否越界或剧烈变化。
- 是否完成一整段路径，而不是仅依赖通用 `done`。

### 8.4 依赖版本未固定

`requirements.txt` 当前只列出包名，没有固定本次可运行环境对应的版本。后续应在确认完整 runtime 可用后，决定是否锁定 NumPy/SciPy/scikit-image/CasADi 等关键依赖版本，或建立独立虚拟环境。

### 8.5 文本编码需要留意

部分源码中的中文注释在 PowerShell 输出中显示为乱码。继续修改这些文件前，应确认文件实际编码，避免保存时破坏中文内容或字符串匹配逻辑，特别是 `llm/decomposer.py` 中的中文关键词。

## 9. 建议的后续步骤

1. 为纯 `target_tracking` 执行入口增加可配置的测试步数上限，先运行 2 到 10 步真实 HoloOcean 测试。
2. 在有限步数测试中打印或断言 observation 维度、action 维度以及 MPC 是否成功求解。
3. 再测试包含 RL 和 MPC 的任务序列，确认从 navigation/obstacle avoidance 切换到 target_tracking 时 observation handoff 正确。
4. 明确 waypoints 的数据接口，将规划结果传给 `TargetTrackingEnv`，替代当前仅用于验证的圆形默认路径。
5. 扩充 `TargetTrackingEvaluator` 的轨迹误差指标。
6. 在 runtime 全链路稳定后，整理并固定可复现的 Python 依赖环境。

## 10. 快速定位索引

| 要查看的问题 | 文件 |
| --- | --- |
| MPC 参数、horizon、观测维度 | `src/smart_captain/skills/target_tracking/config.py` |
| 默认圆形路径、MPC observation 构造 | `src/smart_captain/skills/target_tracking/env.py` |
| MPC 控制求解器 | `src/smart_captain/skills/target_tracking/mpc_controller.py` |
| policy 调用入口 | `src/smart_captain/skills/target_tracking/policy.py` |
| evaluator 当前逻辑 | `src/smart_captain/skills/target_tracking/evaluator.py` |
| 技能注册 | `src/smart_captain/skills/registry.py` |
| mode `2` 与环境绑定 | `src/smart_captain/app/bridge.py` |
| RL/MPC runtime 选择与 observation handoff | `src/smart_captain/app/compat_runtime.py` |
| 默认真实环境与最大步数 | `src/smart_captain/simulation/defaults.py` |

