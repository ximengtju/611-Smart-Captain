"""Navigation skill environment."""

from __future__ import annotations

import numpy as np

from smart_captain.simulation.compat import BASE_CONFIG, BaseEnvironment
from smart_captain.skills.base import SkillAdapter
from smart_captain.skills.navigation.config import NAVIGATION_SPEC, NavigationConfig


class NavigationEnv(BaseEnvironment, SkillAdapter):
    """Native navigation environment path for the new framework."""

    spec = NAVIGATION_SPEC

    def __init__(self, env_config: dict = BASE_CONFIG, auv=None, train_mode: bool = True):
        self.skill_config = NavigationConfig(
            r_min=env_config.get("r_min", 30.0),
            r_max=env_config.get("r_max", 40.0),
            z_min=-15.0,
            z_max=-5.0,
        )
        SkillAdapter.__init__(self, runtime_env=auv)
        BaseEnvironment.__init__(self, env_config, auv, train_mode)
        self.r_min = self.skill_config.r_min
        self.r_max = self.skill_config.r_max

    def sample_goal(self, auv_yaw: float) -> np.ndarray:
        """Sample a navigation target using the migrated task1 logic."""
        r_target = np.random.uniform(self.skill_config.r_min, self.skill_config.r_max)
        psi_target_rel = np.random.uniform(-np.pi, np.pi)
        psi_target_abs = self.ssa(auv_yaw + psi_target_rel)
        x_target = r_target * np.cos(psi_target_abs)
        y_target = r_target * np.sin(psi_target_abs)
        z_target = np.random.uniform(self.skill_config.z_min, self.skill_config.z_max)
        return np.array([x_target, y_target, z_target], dtype=np.float32)

    def generate_environment(self, manual_import: bool = False, import_location=None):
        """Set or sample the next navigation goal."""
        super().generate_environment()

        if manual_import is False:
            self.goal_location = self.sample_goal(self.auv_attitude[2])
        else:
            self.goal_location = np.asarray(import_location, dtype=np.float32)
