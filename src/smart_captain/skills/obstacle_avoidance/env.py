"""Task4 obstacle-avoidance environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from smart_captain.simulation.compat import BASE_CONFIG, BaseEnvironment
from smart_captain.skills.base import SkillAdapter
from smart_captain.skills.obstacle_avoidance.config import (
    OBSTACLE_AVOIDANCE_SPEC,
    ObstacleAvoidanceConfig,
)
from smart_captain.skills.obstacle_avoidance.reward import compute_task4_reward


class ObstacleAvoidanceEnv(BaseEnvironment, SkillAdapter):
    """OpenWater Task4 obstacle-avoidance skill.

    The migrated Task4 contract is:
    - observation: 37 dims, `[navigation state 13] + [compressed rangefinder 24]`
    - action: 4 dims, `[surge, sway, heave, yaw]`
    - world: OpenWater
    """

    spec = OBSTACLE_AVOIDANCE_SPEC

    def __init__(self, env_config: dict[str, Any] = BASE_CONFIG, auv=None, train_mode: bool = True):
        self.skill_config = ObstacleAvoidanceConfig.from_env_config(env_config)
        SkillAdapter.__init__(self, runtime_env=auv)
        BaseEnvironment.__init__(self, env_config, auv, train_mode)
        self._configure_observation_space()

    def _configure_observation_space(self) -> None:
        """Expose the Task4 observation space expected by the trained policy."""
        obs_low = -np.ones(self.skill_config.observation_dim, dtype=np.float32)
        obs_low[0] = 0.0
        obs_low[13:] = 0.0
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=np.ones(self.skill_config.observation_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def sample_goal(self, auv_yaw: float) -> np.ndarray:
        """Sample a Task4 target around the current AUV pose."""
        r_target = np.random.uniform(self.skill_config.r_min, self.skill_config.r_max)
        psi_target_rel = np.random.uniform(-np.pi, np.pi)
        psi_target_abs = self.ssa(auv_yaw + psi_target_rel)

        x_target = self.auv_position[0] + r_target * np.cos(psi_target_abs)
        y_target = self.auv_position[1] + r_target * np.sin(psi_target_abs)
        z_target = self.auv_position[2] + np.random.uniform(
            self.skill_config.z_offset_min,
            self.skill_config.z_offset_max,
        )

        return np.array([x_target, y_target, z_target], dtype=np.float32)

    def generate_environment(self, manual_import: bool = False, import_location=None) -> None:
        """Set the current episode goal."""
        if manual_import:
            self.goal_location = np.asarray(import_location, dtype=np.float32)
        elif self.external_goal is not None:
            self.goal_location = self.external_goal.copy()
        else:
            self.goal_location = self.sample_goal(self.auv_attitude[2])
        self.heading_goal_reached = 0.0

    def reset(
        self,
        seed=None,
        return_info: bool = True,
        options=None,
        manual_import: bool = False,
        import_location=None,
    ):
        """Reset the Task4 skill.

        The compatibility adapter calls Gymnasium-style reset arguments. Legacy
        calls of `reset(manual_import, import_location)` are still accepted.
        """
        if isinstance(seed, bool):
            manual_import = seed
            import_location = return_info
            seed = None
            return_info = True
            options = None

        observation, info = super().reset(seed=seed, return_info=True, options=options)

        if manual_import:
            self.generate_environment(manual_import=True, import_location=import_location)
            self._refresh_goal_dependent_state()
            observation = self.observe()

        if return_info:
            return observation, info
        return observation

    def _refresh_goal_dependent_state(self) -> None:
        """Refresh state caches after the goal is changed externally."""
        errors = self.update_navigation_errors()
        self.delta_d_list = [errors["delta_d"]]
        self.episode_state.delta_d_history = self.delta_d_list
        diff = self.goal_location - self.auv_position
        self.delta_d_xy_last = float(np.linalg.norm(diff[:2]))
        self.delta_d_init = float(errors["delta_d"])
        self.z_start = float(self.auv_position[2])

    def reward_step_impro(self, action: np.ndarray) -> float:
        """Compute the Task4 reward through the skill reward module."""
        return compute_task4_reward(self, action)


__all__ = ["ObstacleAvoidanceEnv"]
