"""Skill registry placeholder.

The registry is the extension point for adding new RL-executable subtasks
without rewriting the application entrypoint.
"""

from smart_captain.skills.base import SkillSpec
from smart_captain.skills.mapping.env import MappingEnv
from smart_captain.skills.navigation.config import NAVIGATION_SPEC
from smart_captain.skills.obstacle_avoidance.config import OBSTACLE_AVOIDANCE_SPEC
from smart_captain.skills.search.env import SearchEnv
from smart_captain.skills.target_tracking.env import TargetTrackingEnv
from smart_captain.skills.target_tracking.config import TARGET_TRACKING_SPEC

SKILL_REGISTRY: dict[str, SkillSpec] = {
    NAVIGATION_SPEC.name: NAVIGATION_SPEC,
    OBSTACLE_AVOIDANCE_SPEC.name: OBSTACLE_AVOIDANCE_SPEC,
    TARGET_TRACKING_SPEC.name: TARGET_TRACKING_SPEC,
    
    "search": SkillSpec(
        name="search",
        env_cls="smart_captain.skills.search.env:SearchEnv",
        policy_cls="smart_captain.skills.search.policy:SearchPolicy",
        default_scenario="dam",
        observation_dim=37,
        action_dim=4,
        description="Placeholder search skill.",
        default_sensors=("dynamics", "velocity", "imaging_sonar"),
        tags=("search", "perception"),
    ),
    "mapping": SkillSpec(
        name="mapping",
        env_cls="smart_captain.skills.mapping.env:MappingEnv",
        policy_cls="smart_captain.skills.mapping.policy:MappingPolicy",
        default_scenario="dam",
        observation_dim=37,
        action_dim=4,
        description="Placeholder mapping skill.",
        default_sensors=("dynamics", "velocity", "imaging_sonar"),
        tags=("mapping", "perception"),
    ),
}


__all__ = ["SKILL_REGISTRY", "SkillSpec", "TargetTrackingEnv", "SearchEnv", "MappingEnv"]
