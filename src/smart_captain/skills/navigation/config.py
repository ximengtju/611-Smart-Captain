"""Navigation skill configuration extracted from the legacy task design."""

from __future__ import annotations

from dataclasses import dataclass

from smart_captain.skills.base import SkillSpec


@dataclass(frozen=True)
class NavigationConfig:
    """Config values that define navigation-goal sampling behavior."""

    r_min: float = 30.0
    r_max: float = 40.0
    z_min: float = -15.0
    z_max: float = -5.0


NAVIGATION_SPEC = SkillSpec(
    name="navigation",
    env_cls="smart_captain.skills.navigation.env:NavigationEnv",
    policy_cls="smart_captain.skills.navigation.policy:NavigationPolicy",
    default_scenario="pier_harbor",
    observation_dim=37,
    action_dim=4,
    description="Waypoint navigation in underwater scenes using base kinematics and range sensors.",
    default_sensors=("dynamics", "velocity", "rangefinder"),
    tags=("locomotion", "waypoint", "baseline"),
    train_entrypoint="smart_captain.skills.navigation.train",
    reward_entrypoint="smart_captain.skills.navigation.reward",
    config_entrypoint="smart_captain.skills.navigation.config:NAVIGATION_SPEC",
)
