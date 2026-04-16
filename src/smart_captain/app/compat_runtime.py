"""Compatibility runtime that combines the new planner with the legacy stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smart_captain.app.bridge import LegacyExecutionPlan, LegacyModeBridge
from smart_captain.app.shared_auv_runtime import create_shared_auv_runtime
from smart_captain.rl.agents import LegacyCompatibleAgents
from smart_captain.simulation.defaults import DEFAULT_ENV_CONFIG


@dataclass(frozen=True)
class LegacyRuntimeConfig:
    """Minimal config required to drive the legacy runtime with a new plan."""

    model_paths: list[str] = field(default_factory=lambda: [
        "models/rl/navigation/sac/task1-v0_SAC_3/task1-v0_SAC_400000.zip",
        "models/rl/obstacle_avoidance/sac/task2-v0_SAC_1/task2-v0_SAC_200000.zip",
    ])
    model_types: list[str] = field(default_factory=lambda: ["sac", "sac"])
    mission_id: str = "legacy_runtime_mission"
    command: str = "请控制水下机器人先导航到目标区域，途中避障，然后搜索可疑目标，最后切换声呐进行精细测绘"
    world_state: dict[str, Any] = field(default_factory=dict)


def build_legacy_plan(config: LegacyRuntimeConfig | None = None) -> LegacyExecutionPlan:
    """Build a new-style mission plan for the legacy runtime."""
    config = config or LegacyRuntimeConfig()
    bridge = LegacyModeBridge()
    return bridge.build_plan(
        command=config.command,
        mission_id=config.mission_id,
        world_state=config.world_state,
    )


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
    agents = LegacyCompatibleAgents(
        model_paths=config.model_paths,
        model_types=config.model_types,
        model_class_dict=model_class_dict,
        mode="multi",
    )
    return auv, agents, plan


def create_adapter_backed_legacy_runtime(config: LegacyRuntimeConfig | None = None):
    """Instantiate the legacy runtime using the new shared-AUV adapter.

    This still depends on legacy task classes and HoloOcean, but the runtime
    composition is now expressed inside `src/smart_captain/` instead of being
    hard-coded only in `task_combine1.py`.
    """
    config = config or LegacyRuntimeConfig()

    from stable_baselines3 import A2C, PPO, SAC

    plan = build_legacy_plan(config)
    runtime = create_shared_auv_runtime(
        layout=plan.runtime_layout,
        env_config=DEFAULT_ENV_CONFIG,
        mode=0,
        show_viewport=True,
    )
    model_class_dict = {"sac": SAC, "a2c": A2C, "ppo": PPO}
    agents = LegacyCompatibleAgents(
        model_paths=config.model_paths,
        model_types=config.model_types,
        model_class_dict=model_class_dict,
        mode="multi",
    )
    return runtime.adapter, agents, plan


def execute_adapter_backed_legacy_plan(config: LegacyRuntimeConfig | None = None) -> dict[str, Any]:
    """Run the mission loop using the new shared-AUV adapter composition."""
    config = config or LegacyRuntimeConfig()
    auv, agents, plan = create_adapter_backed_legacy_runtime(config)

    if not plan.mode_sequence:
        return {
            "mission_id": plan.mission_id,
            "skills": plan.skill_sequence,
            "mode_sequence": [],
            "completed_subtasks": [],
            "final_info": {},
            "runtime": "adapter-backed-legacy",
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
        "runtime": "adapter-backed-legacy",
    }


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
