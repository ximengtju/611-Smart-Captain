"""Task1 navigation skill configuration.

This module is the single home for migrated Task1 skill metadata, target
sampling settings, and default SAC hyper-parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_captain.skills.base import SkillSpec


@dataclass(frozen=True)
class NavigationConfig:
    """Task1 observation shape and navigation-goal sampling behavior."""

    observation_dim: int = 37
    action_dim: int = 4
    r_min: float = 40.0
    r_max: float = 60.0
    z_min: float = -20.0
    z_max: float = 20.0
    default_scenario: str = "pier_harbor"

    @classmethod
    def from_env_config(cls, env_config: dict[str, Any]) -> "NavigationConfig":
        """Create skill config from the shared environment config."""
        return cls(
            observation_dim=env_config.get("n_observations", cls.observation_dim),
            action_dim=env_config.get("n_actions", cls.action_dim),
            r_min=env_config.get("r_min", cls.r_min),
            r_max=env_config.get("r_max", cls.r_max),
            z_min=env_config.get("task1.z_min", cls.z_min),
            z_max=env_config.get("task1.z_max", cls.z_max),
        )


TASK1_SAC_HYPER_PARAMS_DEFAULT = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 100,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
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

TASK1_SAC_HYPER_PARAMS = {
    "learning_rate": 0.0015,
    "buffer_size": 50_000,
    "learning_starts": 100,
    "batch_size": 128,
    "tau": 0.005,
    "gamma": 0.99,
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


DEFAULT_HYPER_PARAMS = TASK1_SAC_HYPER_PARAMS


NAVIGATION_SPEC = SkillSpec(
    name="navigation",
    env_cls="smart_captain.skills.navigation.env:NavigationEnv",
    policy_cls="smart_captain.skills.navigation.policy:NavigationPolicy",
    default_scenario="pier_harbor",
    observation_dim=37,
    action_dim=4,
    description=(
        "Task1 PierHarbor waypoint navigation with 37-dimensional state/range "
        "observations and 4-axis HoveringAUV control."
    ),
    default_sensors=("dynamics", "velocity", "rangefinder"),
    tags=("task1", "pier_harbor", "locomotion", "waypoint", "baseline"),
    train_entrypoint="smart_captain.skills.navigation.train",
    reward_entrypoint="smart_captain.skills.navigation.reward",
    config_entrypoint="smart_captain.skills.navigation.config:NAVIGATION_SPEC",
    metadata={
        "legacy_source": "F:/HoloOcean-release/BaseEnv/Env_Task1",
        "algorithm": "sac",
        "gym_id": "task1-v0",
        "action_order": ("surge", "sway", "heave", "yaw"),
    },
)


__all__ = [
    "DEFAULT_HYPER_PARAMS",
    "NAVIGATION_SPEC",
    "TASK1_SAC_HYPER_PARAMS",
    "TASK1_SAC_HYPER_PARAMS_DEFAULT",
    "NavigationConfig",
]
