"""Compatibility runtime that combines the new planner with the legacy stack."""
# 兼容运行配置层
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from smart_captain.app.bridge import LegacyExecutionPlan, LegacyModeBridge
from smart_captain.app.shared_auv_runtime import create_shared_auv_runtime
from smart_captain.configs.mission_positions import (
    NAV_GOAL,
    NAV_ROTATION,
    NAV_START,
    OBSTACLE_START,
    OBSTACLE_GOAL,
)
from smart_captain.orchestration.dispatcher import TaskDispatcher
from smart_captain.orchestration.evaluator_registry import evaluate_skill_progress
from smart_captain.orchestration.feedback_adapter import build_feedback_snapshot
from smart_captain.orchestration.recovery import DecisionAction
from smart_captain.orchestration.task_graph import TaskStatus

from smart_captain.rl.agents import LegacyCompatibleAgents
from smart_captain.simulation.defaults import DEFAULT_ENV_CONFIG

from smart_captain.skills.target_tracking.policy import TargetTrackingPolicy

@dataclass(frozen=True)
# LegacyRuntimeConfig 是一个兼容旧运行链的配置对象
# 把运行旧式多模型/多 mode 任务链需要的东西集中放在一起，比如模型路径、模型类型、任务指令、任务 ID、世界状态。
class LegacyRuntimeConfig:
    """Minimal config required to drive the legacy runtime with a new plan."""

    model_paths: list[str] = field(default_factory=lambda: [
        #r"models\rl\navigation\sac\task1-v0_SAC_1\task1-v0_SAC_1500000.zip",
        r"models\rl\navigation\sac\task1-v0_SAC_2\task1-v0_SAC_2000000.zip",
        "models/rl/obstacle_avoidance/sac/task4-v0_SAC_1/task4-v0_SAC_3200000.zip",
    ])
    model_types: list[str] = field(default_factory=lambda: ["sac", "sac"])
    mission_id: str = "legacy_runtime_mission"
    command: str = "请控制水下机器人先导航到目标区域，途中避障，然后搜索可疑目标，最后切换声呐进行精细测绘"
    world_state: dict[str, Any] = field(default_factory=dict)


#根据配置里的自然语言任务命令，生成一个旧版运行时可以执行的任务计划 LegacyExecutionPlan
#把自然语言 command 解析成任务图和 mode_sequence
def build_legacy_plan(config: LegacyRuntimeConfig | None = None) -> LegacyExecutionPlan:
    """Build a new-style mission plan for the legacy runtime."""
    config = config or LegacyRuntimeConfig()
    #创建一个桥接器。这个桥接器负责把新系统的任务图转换成旧系统能理解的整数 mode 序列。
    bridge = LegacyModeBridge()
    return bridge.build_plan(
        command=config.command,
        mission_id=config.mission_id,
        world_state=config.world_state,
    )

#MPC新增兼容部分
class MixedSkillAgents:
    def __init__(self, rl_agents, extra_policies):
        self.rl_agents = rl_agents
        self.extra_policies = extra_policies
        self.current_multi_mode_index = 0

    def set_multi_mode_index(self, index: int) -> None:
        self.current_multi_mode_index = index
        if index not in self.extra_policies:
            self.rl_agents.set_multi_mode_index(index)

    def predict(self, obs, state=None, deterministic: bool = True):
        if self.current_multi_mode_index in self.extra_policies:
            return self.extra_policies[self.current_multi_mode_index].predict(
                obs,
                state=state,
                deterministic=deterministic,
            )
        return self.rl_agents.predict(
            obs,
            state=state,
            deterministic=deterministic,
        )

    @property
    def active_skill(self):
        return getattr(self.rl_agents, "active_skill", None)

#MPC新增兼容部分结束

def preview_legacy_plan(config: LegacyRuntimeConfig | None = None) -> dict[str, Any]:
    """Return a serializable summary without touching the simulator runtime."""
    config = config or LegacyRuntimeConfig()
    plan = build_legacy_plan(config)
    return {
        "mission_id": plan.mission_id,
        "command": plan.command,
        "skills": plan.skill_sequence,
        "mode_sequence": plan.mode_sequence,
        "subtask_ids": plan.subtask_ids,
        "mode_to_skill": plan.runtime_layout.mode_to_skill() if plan.runtime_layout else {},
        "task_bindings": [
            {
                "mode_index": binding.mode_index,
                "skill_name": binding.skill_name,
                "env_import_path": binding.env_import_path,
            }
            for binding in (plan.runtime_layout.task_bindings if plan.runtime_layout else [])
        ],
        "model_paths": config.model_paths,
    }


def _build_fixed_start_env_config() -> dict[str, Any]:
    """Apply mission_positions.py start pose to a copied runtime config."""
    env_config = copy.deepcopy(DEFAULT_ENV_CONFIG)
    env_config["auv_config"]["agents"][0]["location"] = NAV_START
    env_config["auv_config"]["agents"][0]["rotation"] = NAV_ROTATION
    return env_config


def _fixed_goal_for_skill(skill: str):
    """Return the configured goal for skills with fixed mission endpoints."""
    if skill == "navigation":
        return NAV_GOAL
    if skill == "obstacle_avoidance":
        return OBSTACLE_GOAL
    return None


def _set_fixed_goal_for_skill(active_env, skill: str, goal_override=None) -> None:
    """Set the active environment goal when the mission config defines one."""
    goal = goal_override if goal_override is not None else _fixed_goal_for_skill(skill)
    if goal is not None:
        active_env.set_goal(goal)


def _mode_for_skill(plan: LegacyExecutionPlan, skill: str) -> int:
    """Resolve the legacy mode index that runs a skill."""
    if plan.runtime_layout is not None:
        skill_to_mode = plan.runtime_layout.skill_to_mode()
        if skill in skill_to_mode:
            return skill_to_mode[skill]
    if skill in plan.skill_sequence:
        return plan.mode_sequence[plan.skill_sequence.index(skill)]
    raise KeyError(f"No runtime mode is registered for skill '{skill}'")


def _to_jsonable(value):
    """Convert runtime values to objects accepted by `json.dumps`.

    HoloOcean/Gym feedback often carries NumPy arrays and NumPy scalar types in
    `info`, especially fields such as `position`.  The mission runner prints the
    returned summary as JSON, so runtime summaries must cross this boundary as
    plain dict/list/str/int/float/bool values.
    """
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prepare_active_env_after_switch(active_env, skill: str, goal_override=None):
    """Continue from the previous skill state without resetting HoloOcean."""
    active_env.done = False
    active_env.goal_reached = False
    if hasattr(active_env, "episode_state"):
        active_env.episode_state.done = False
        active_env.episode_state.goal_reached = False

    _set_fixed_goal_for_skill(active_env, skill, goal_override=goal_override)
    active_env.generate_environment()
    active_env.update_navigation_errors()
    active_env.delta_d_list = [active_env.delta_d]
    if hasattr(active_env, "episode_state"):
        active_env.episode_state.delta_d_history = active_env.delta_d_list
    #MPC兼容修改
    if skill == "target_tracking" and hasattr(active_env, "build_tracking_observation"):
        obs = active_env.build_tracking_observation()
    else:
        obs = active_env.observe()
    #MPC兼容修改结束
    info = getattr(active_env, "info", {})
    return obs, info


def _feedback_record_interval(config: LegacyRuntimeConfig) -> int:
    """Return how often non-event feedback should be saved to history.

    Feedback is still evaluated every step.  This interval only controls the
    size of saved histories and JSON artifacts.
    """
    mission_state = config.world_state.get("mission", {})
    try:
        interval = int(mission_state.get("feedback_record_interval", 50))
    except Exception:
        interval = 50
    return max(1, interval)


def create_legacy_runtime(config: LegacyRuntimeConfig | None = None):
    """Instantiate legacy env and agents while sourcing the plan from the new stack.

    This mirrors the old mode-switching runtime without importing
    `task.task_combine1.MAInterface` from the legacy framework.
    """
    config = config or LegacyRuntimeConfig()

    from stable_baselines3 import A2C, PPO, SAC

    plan = build_legacy_plan(config)
    auv = create_shared_auv_runtime(
        layout=plan.runtime_layout,
        env_config=DEFAULT_ENV_CONFIG,
        mode=0,
        show_viewport=True,
    )
    model_class_dict = {"sac": SAC, "a2c": A2C, "ppo": PPO}
    rl_agents = LegacyCompatibleAgents(         #变量名agents——>rl_agents
        model_paths=config.model_paths,
        model_types=config.model_types,
        model_class_dict=model_class_dict,
        mode="multi",
    )

    #MPC兼容部分
    agents = MixedSkillAgents(
    rl_agents=rl_agents,
    extra_policies={
        2: TargetTrackingPolicy(),
    },
    )
    #MPC兼容部分结束


    return auv, agents, plan


# 创建共享 AUV、多个 env、多个 RL model
def create_adapter_backed_legacy_runtime(config: LegacyRuntimeConfig | None = None):

    config = config or LegacyRuntimeConfig()

    from stable_baselines3 import A2C, PPO, SAC

    #根据配置构建任务计划 ，command 变成任务图
    plan = build_legacy_plan(config)

    env_config = _build_fixed_start_env_config()

    #创建一个共享 HoloOcean AUV
    runtime = create_shared_auv_runtime(
        layout=plan.runtime_layout,
        env_config=env_config,
        mode=0,
        show_viewport=True,
    )
    model_class_dict = {"sac": SAC, "a2c": A2C, "ppo": PPO}

    #加载两个 RL 模型
    rl_agents = LegacyCompatibleAgents(         #变量名agents——>rl_agents
        model_paths=config.model_paths,
        model_types=config.model_types,
        model_class_dict=model_class_dict,
        mode="multi",
    )

    #MPC兼容部分
    agents = MixedSkillAgents(
    rl_agents=rl_agents,
    extra_policies={
        2: TargetTrackingPolicy(),
    },
    )
    #MPC兼容部分结束

    return runtime.adapter, agents, plan


def execute_adapter_backed_legacy_plan(config: LegacyRuntimeConfig | None = None) -> dict[str, Any]:
    config = config or LegacyRuntimeConfig()
    # 创建三个东西
    # plan 任务计划，auv - HoloOcean + 技能环境适配器，agents-RL模型管理器
    auv, agents, plan = create_adapter_backed_legacy_runtime(config)

    if not plan.mode_sequence:
        return _to_jsonable({
            "mission_id": plan.mission_id,
            "skills": plan.skill_sequence,
            "mode_sequence": [],
            "completed_subtasks": [],
            "final_info": {},
            "runtime": "adapter-backed-legacy",
        })

    dispatcher = TaskDispatcher()
    context = dispatcher.bootstrap_context(plan.graph, initial_world_state=config.world_state)
    feedback_record_interval = _feedback_record_interval(config)

    current_subtask_idx = 0
    total_subtasks = len(plan.mode_sequence)
    completed_subtasks: list[str] = []
    final_info: dict[str, Any] = {}
    step_count = 0

    first_mode = plan.mode_sequence[0]
    auv.set_multi_mode_index(first_mode)
    agents.set_multi_mode_index(first_mode)

    active_env = auv.task_env[auv.mode]
    _set_fixed_goal_for_skill(active_env, plan.skill_sequence[0])

    obs, info = auv.reset()

    final_info = info
    dispatcher.step(plan.graph, context)
    dispatcher.update_world_state(
        context,
        {
            "mission": {
                "active_controller_skill": plan.skill_sequence[0],
                "controller_skill": plan.skill_sequence[0],
                "task_skill": plan.skill_sequence[0],
            }
        },
    )

    while current_subtask_idx < total_subtasks:
        done = False
        subtask = plan.graph.subtasks[current_subtask_idx]
        while not done:
            action, _ = agents.predict(obs)
            obs, reward, done, truncated, info = auv.step(action)
            final_info = info
            step_count += 1

            active_env = auv.task_env[auv.mode]
            controller_skill = (
                context.world_state
                .get("mission", {})
                .get("active_controller_skill", subtask.skill)
            )
            feedback = build_feedback_snapshot(
                mission_id=plan.mission_id,
                subtask_id=subtask.id,
                skill=subtask.skill,
                step_count=step_count,
                observation=obs,
                reward=reward,
                done=done,
                truncated=truncated,
                info=info,
                active_env=active_env,
                controller_skill=controller_skill,
            )
            progress = evaluate_skill_progress(
                subtask=subtask,
                context=context,
                feedback=feedback,
            )
            is_key_event = (
                progress.status.value != "running"
                or done
                or truncated
                or bool(feedback.collision)
                or bool(feedback.goal_reached)
            )
            record_history = is_key_event or (step_count % feedback_record_interval == 0)
            decision = dispatcher.step_with_progress(
                graph=plan.graph,
                context=context,
                progress=progress,
                feedback=feedback,
                record_history=record_history,
            )
            final_info = {
                **final_info,
                "task_skill": feedback.task_skill,
                "controller_skill": feedback.controller_skill,
                "progress": progress.to_record(),
                "decision": decision.to_record(),
            }

            if decision.action == DecisionAction.ADVANCE:
                done = True
            elif decision.action == DecisionAction.SWITCH_SKILL:
                details = decision.details or {}
                target_skill = details.get("requested_skill")
                if target_skill is None:
                    continue

                preserved_goal = None
                if details.get("preserve_goal", False):
                    preserved_goal = feedback.goal
                    if preserved_goal is None:
                        preserved_goal = getattr(active_env, "goal_location", None)

                target_mode = _mode_for_skill(plan, target_skill)
                if auv.mode != target_mode:
                    auv.set_multi_mode_index(target_mode)
                    agents.set_multi_mode_index(target_mode)

                active_env = auv.task_env[auv.mode]
                obs, info = _prepare_active_env_after_switch(
                    active_env,
                    target_skill,
                    goal_override=preserved_goal,
                )
                final_info = {
                    **getattr(active_env, "info", {}),
                    "task_skill": subtask.skill,
                    "controller_skill": target_skill,
                    "progress": progress.to_record(),
                    "decision": decision.to_record(),
                }
                dispatcher.update_world_state(
                    context,
                    {
                        "mission": {
                            "active_controller_skill": target_skill,
                            "controller_skill": target_skill,
                            "task_skill": subtask.skill,
                        }
                    },
                )
                controller_skill = target_skill
                print(
                    f"[reactive-switch] subtask={subtask.skill} "
                    f"controller={target_skill} goal=preserved"
                )
            elif decision.action in {DecisionAction.REQUEST_REPLAN, DecisionAction.ABORT}:
                return _to_jsonable({
                    "mission_id": plan.mission_id,
                    "skills": plan.skill_sequence,
                    "mode_sequence": plan.mode_sequence,
                    "completed_subtasks": context.completed_subtasks,
                    "failed_subtasks": context.failed_subtasks,
                    "replan_requested": context.replan_requested,
                    "replan_reason": context.replan_reason,
                    "final_info": final_info,
                    "world_state": context.world_state,
                    "progress_by_subtask": context.progress_by_subtask,
                    "feedback_record_interval": feedback_record_interval,
                    "feedback_tail": context.feedback_history[-10:],
                    "decisions": context.decisions,
                    "runtime": "adapter-backed-legacy",
                })

            if step_count % 100 == 0 and hasattr(auv, "task_env"):
                active_env = auv.task_env[auv.mode]
                goal = getattr(active_env, "goal_location", None)
                position = getattr(active_env, "auv_position", None)
                velocity = getattr(active_env, "auv_relative_velocity", None)
                if goal is not None and position is not None:
                    goal_arr = np.asarray(goal, dtype=np.float32)
                    position_arr = np.asarray(position, dtype=np.float32)
                    distance = float(np.linalg.norm(goal_arr - position_arr))
                    speed = (
                        float(np.linalg.norm(np.asarray(velocity, dtype=np.float32)))
                        if velocity is not None
                        else float("nan")
                    )
                    nearest_text = (
                        f"{feedback.nearest_obstacle_distance:.2f}"
                        if feedback.nearest_obstacle_distance is not None
                        else "nan"
                    )
                    print(
                        f"[step {step_count}] "
                        f"task={subtask.skill} "
                        f"model={controller_skill} "
                        f"nearest_obs={nearest_text} "
                        f"goal={goal_arr.round(1).tolist()} "
                        f"position={position_arr.round(1).tolist()} "
                        f"distance={distance:.1f} "
                        f"speed={speed:.3f}"
                    )

        if subtask.status == TaskStatus.SUCCEEDED and subtask.id not in completed_subtasks:
            completed_subtasks.append(subtask.id)
        current_subtask_idx += 1
        if current_subtask_idx < total_subtasks:
            # next_mode = plan.mode_sequence[current_subtask_idx]
            # auv.set_multi_mode_index(next_mode)
            # agents.set_multi_mode_index(next_mode)
            # obs, info = auv.reset()
            # final_info = info
            next_mode = plan.mode_sequence[current_subtask_idx]
            next_skill = plan.skill_sequence[current_subtask_idx]
            auv.set_multi_mode_index(next_mode)
            agents.set_multi_mode_index(next_mode)

            active_env = auv.task_env[auv.mode]
            print(
                f"[handoff] from={subtask.skill} to={next_skill} "
                "mode=continuous"
            )
            obs, info = _prepare_active_env_after_switch(active_env, next_skill)
            final_info = {
                **info,
                "task_skill": next_skill,
                "controller_skill": next_skill,
            }
            dispatcher.update_world_state(
                context,
                {
                    "mission": {
                        "active_controller_skill": next_skill,
                        "controller_skill": next_skill,
                        "task_skill": next_skill,
                    }
                },
            )

    return _to_jsonable({
        "mission_id": plan.mission_id,
        "skills": plan.skill_sequence,
        "mode_sequence": plan.mode_sequence,
        "completed_subtasks": completed_subtasks,
        "failed_subtasks": context.failed_subtasks,
        "replan_requested": context.replan_requested,
        "replan_reason": context.replan_reason,
        "final_info": final_info,
        "world_state": context.world_state,
        "progress_by_subtask": context.progress_by_subtask,
        "feedback_record_interval": feedback_record_interval,
        "feedback_tail": context.feedback_history[-10:],
        "decisions": context.decisions,
        "runtime": "adapter-backed-legacy",
    })


def execute_legacy_plan(config: LegacyRuntimeConfig | None = None) -> dict[str, Any]:
    """Run the legacy multi-model loop using a plan from the new planner.

    This mirrors the control flow of `Main-Framework/main.py` while keeping the
    legacy files untouched.
    """
    config = config or LegacyRuntimeConfig()
    auv, agents, plan = create_legacy_runtime(config)

    if not plan.mode_sequence:
        return {
            "mission_id": plan.mission_id,
            "skills": plan.skill_sequence,
            "mode_sequence": [],
            "completed_subtasks": [],
            "final_info": {},
        }

    current_subtask_idx = 0
    total_subtasks = len(plan.mode_sequence)
    completed_subtasks: list[str] = []
    final_info: dict[str, Any] = {}

    first_mode = plan.mode_sequence[0]
    auv.set_multi_mode_index(first_mode)
    agents.set_multi_mode_index(first_mode)
    obs, info = auv.reset()
    final_info = info

    while current_subtask_idx < total_subtasks:
        done = False
        while not done:
            action, _ = agents.predict(obs)
            obs, reward, done, truncated, info = auv.step(action)
            final_info = info

        completed_subtasks.append(plan.subtask_ids[current_subtask_idx])
        current_subtask_idx += 1
        if current_subtask_idx < total_subtasks:
            next_mode = plan.mode_sequence[current_subtask_idx]
            auv.set_multi_mode_index(next_mode)
            agents.set_multi_mode_index(next_mode)
            obs, info = auv.reset()
            final_info = info

    return {
        "mission_id": plan.mission_id,
        "skills": plan.skill_sequence,
        "mode_sequence": plan.mode_sequence,
        "completed_subtasks": completed_subtasks,
        "final_info": final_info,
    }
