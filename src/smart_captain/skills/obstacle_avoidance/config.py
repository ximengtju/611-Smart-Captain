"""Obstacle-avoidance skill configuration extracted from legacy task2."""

from __future__ import annotations

from dataclasses import dataclass

from smart_captain.skills.base import SkillSpec


@dataclass(frozen=True)
class ObstacleAvoidanceConfig:
    """Legacy task2 observation shaping and target sampling settings."""

    observation_dim: int = 40
    reduced_horizontal_indices: tuple[int, ...] = (20, 21, 22, 23, 0, 1, 2, 3, 4)
    vertical_reduce_stride: int = 3
    r_min: float = 30.0
    r_max: float = 40.0
    z_min: float = -15.0
    z_max: float = -5.0


OBSTACLE_AVOIDANCE_SPEC = SkillSpec(
    name="obstacle_avoidance",
    env_cls="smart_captain.skills.obstacle_avoidance.env:ObstacleAvoidanceEnv",
    policy_cls="smart_captain.skills.obstacle_avoidance.policy:ObstacleAvoidancePolicy",
    default_scenario="open_water",
    observation_dim=40,
    action_dim=4,
    description="Navigation with obstacle-aware observation reduction and obstacle penalty shaping.",
    default_sensors=("dynamics", "velocity", "rangefinder"),
    tags=("locomotion", "avoidance", "safety"),
    train_entrypoint="smart_captain.skills.obstacle_avoidance.train",
    reward_entrypoint="smart_captain.skills.obstacle_avoidance.reward",
    config_entrypoint="smart_captain.skills.obstacle_avoidance.config:OBSTACLE_AVOIDANCE_SPEC",
)
