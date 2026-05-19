"""Task4 obstacle-avoidance skill configuration.

This module is the single home for the migrated Task4 skill metadata, target
sampling settings, and default SAC hyper-parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_captain.skills.base import SkillSpec


@dataclass(frozen=True)
class ObstacleAvoidanceConfig:
    """Task4 observation shape and target sampling settings."""

    observation_dim: int = 37
    action_dim: int = 4
    r_min: float = 110.0
    r_max: float = 125.0
    z_offset_min: float = -5.0
    z_offset_max: float = 10.0
    default_scenario: str = "open_water"

    @classmethod
    def from_env_config(cls, env_config: dict[str, Any]) -> "ObstacleAvoidanceConfig":
        """Create skill config from the shared environment config."""
        return cls(
            observation_dim=env_config.get("obstacle_observation_dim", cls.observation_dim),
            action_dim=env_config.get("n_actions", cls.action_dim),
            r_min=env_config.get("r_min", cls.r_min),
            r_max=env_config.get("r_max", cls.r_max),
            z_offset_min=env_config.get("task4.z_offset_min", cls.z_offset_min),
            z_offset_max=env_config.get("task4.z_offset_max", cls.z_offset_max),
        )


TASK4_SAC_HYPER_PARAMS = {
    "learning_rate": 0.001,
    "buffer_size": 300_000,
    "learning_starts": 5000,
    "batch_size": 128,
    "tau": 0.005,
    "gamma": 0.997,
    "train_freq": 1,
    "gradient_steps": 1,
    "action_noise": None,
    "replay_buffer_class": None,
    "replay_buffer_kwargs": None,
    "optimize_memory_usage": False,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "tensorboard_log": None,
    "create_eval_env": False,
    "policy_kwargs": None,
    "verbose": 0,
    "seed": None,
    "device": "cuda:0",
    "_init_setup_model": True,
}


OBSTACLE_AVOIDANCE_SPEC = SkillSpec(
    name="obstacle_avoidance",
    env_cls="smart_captain.skills.obstacle_avoidance.env:ObstacleAvoidanceEnv",
    policy_cls="smart_captain.skills.obstacle_avoidance.policy:ObstacleAvoidancePolicy",
    default_scenario="open_water",
    observation_dim=37,
    action_dim=4,
    description=(
        "Task4 OpenWater obstacle-avoidance navigation with 37-dimensional "
        "state/range observations and 4-axis HoveringAUV control."
    ),
    default_sensors=("dynamics", "velocity", "rangefinder"),
    tags=("task4", "open_water", "locomotion", "avoidance", "safety"),
    train_entrypoint="smart_captain.skills.obstacle_avoidance.train",
    reward_entrypoint="smart_captain.skills.obstacle_avoidance.reward",
    config_entrypoint="smart_captain.skills.obstacle_avoidance.config:OBSTACLE_AVOIDANCE_SPEC",
    metadata={
        "legacy_source": "F:/HoloOcean-release/BaseEnv/Env_Task4",
        "algorithm": "sac",
        "gym_id": "task4-v0",
        "action_order": ("surge", "sway", "heave", "yaw"),
    },
)


__all__ = [
    "OBSTACLE_AVOIDANCE_SPEC",
    "TASK4_SAC_HYPER_PARAMS",
    "ObstacleAvoidanceConfig",
]
