"""New-framework runtime entrypoints built on native RL abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smart_captain.app.bridge import LegacyExecutionPlan, LegacyModeBridge
from smart_captain.rl.agents import MultiSkillAgentRuntime
from smart_captain.rl.model_store import ModelStore


@dataclass(frozen=True)
class NativeRuntimeConfig:
    """Minimal mission config for the new-framework runtime facade."""

    mission_id: str = "native_runtime_mission"
    command: str = "请控制水下机器人先导航到目标区域，途中避障，然后搜索可疑目标，最后切换声呐进行精细测绘"
    world_state: dict[str, Any] = field(default_factory=dict)


def build_native_plan(config: NativeRuntimeConfig | None = None) -> LegacyExecutionPlan:
    """Build a mission plan using the new planner stack."""
    config = config or NativeRuntimeConfig()
    bridge = LegacyModeBridge()
    return bridge.build_plan(
        command=config.command,
        mission_id=config.mission_id,
        world_state=config.world_state,
    )


def create_native_agent_runtime() -> MultiSkillAgentRuntime:
    """Create a new-framework multi-skill agent runtime using legacy model files."""
    try:
        from stable_baselines3 import A2C, PPO, SAC
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Native agent runtime requires the legacy Stable Baselines stack and its "
            "dependencies to be installed, including torch."
        ) from exc

    algorithm_loaders = {
        "sac": SAC,
        "a2c": A2C,
        "ppo": PPO,
    }
    return MultiSkillAgentRuntime(
        model_store=ModelStore(),
        algorithm_loaders=algorithm_loaders,
    )


def preview_native_runtime(config: NativeRuntimeConfig | None = None) -> dict[str, Any]:
    """Return a serializable runtime preview without starting the simulator."""
    config = config or NativeRuntimeConfig()
    plan = build_native_plan(config)
    return {
        "mission_id": plan.mission_id,
        "skills": plan.skill_sequence,
        "subtask_ids": plan.subtask_ids,
        "mode_sequence": plan.mode_sequence,
        "model_store": ModelStore().to_summary(),
    }
