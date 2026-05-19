"""Navigation skill environment."""

from __future__ import annotations

import copy

import numpy as np

from smart_captain.simulation.compat import BASE_CONFIG, BaseEnvironment
from smart_captain.skills.base import SkillAdapter
from smart_captain.skills.navigation.config import NAVIGATION_SPEC, NavigationConfig
from smart_captain.skills.navigation.reward import compute_task1_reward


DEFAULT_OPEN_WATER_START = [150, -73, -295]
LEGACY_TASK1_START = [0, 0, -10]
LEGACY_TASK1_ROTATION = [0.0, 0.0, 0.0]

TASK1_CONFIG_OVERRIDES = {
    "u_max": 2.0,
    "v_max": 2.0,
    "w_max": 2.0,
    "max_timesteps": 10000,
    "dist_goal_reached_tol": 1.0,
    "velocity_goal_reached_tol": 0.2,
    "ang_rate_goal_reached_tol": 0.1,
    "attitude_goal_reached_tol": 0.1,
    "max_dist_from_goal": 150.0,
    "r_min": 40.0,
    "r_max": 60.0,
}

TASK1_REWARD_FACTORS = {
    "w_delta_theta": 1.0,
    "w_delta_psi": 1.0,
    "w_phi": 1.0,
    "w_theta": 0.0,
    "w_goal": 60.0,
    "w_deltad_max": -30.0,
    "w_Theta_max": -30.0,
    "w_t_max": -10.0,
    "w_col": -50.0,
}


def task1_navigation_config(env_config: dict) -> dict:
    """Return a navigation config that matches the Task1 training contract."""
    config = copy.deepcopy(env_config)
    config.update(TASK1_CONFIG_OVERRIDES)
    config["reward_factors"] = {
        **config.get("reward_factors", {}),
        **TASK1_REWARD_FACTORS,
    }

    scenario_cfg = config.get("auv_config", {})
    #控制初始化场景
    #scenario_cfg["world"] = "PierHarbor"
    scenario_cfg["world"] = "OpenWater"
    if scenario_cfg.get("agents"):
        agent_cfg = scenario_cfg["agents"][0]
        if agent_cfg.get("location") == DEFAULT_OPEN_WATER_START:
            agent_cfg["location"] = LEGACY_TASK1_START
            agent_cfg["rotation"] = LEGACY_TASK1_ROTATION
    return config


class NavigationEnv(BaseEnvironment, SkillAdapter):
    """Native navigation environment path for the new framework."""

    spec = NAVIGATION_SPEC

    def __init__(self, env_config: dict = BASE_CONFIG, auv=None, train_mode: bool = True):
        env_config = task1_navigation_config(env_config)
        self.skill_config = NavigationConfig.from_env_config(env_config)
        SkillAdapter.__init__(self, runtime_env=auv)
        BaseEnvironment.__init__(self, env_config, auv, train_mode)
        self.r_min = self.skill_config.r_min
        self.r_max = self.skill_config.r_max

    def sample_goal(self, auv_yaw: float) -> np.ndarray:
        r_target = np.random.uniform(self.skill_config.r_min, self.skill_config.r_max)
        psi_target_rel = np.random.uniform(-np.pi, np.pi)
        psi_target_abs = self.ssa(auv_yaw + psi_target_rel)

        x_target = self.auv_position[0] + r_target * np.cos(psi_target_abs)
        y_target = self.auv_position[1] + r_target * np.sin(psi_target_abs)
        z_target = self.auv_position[2] + np.random.uniform(
            self.skill_config.z_min,
            self.skill_config.z_max,
        )

        return np.array([x_target, y_target, z_target], dtype=np.float32)

    def generate_environment(self, manual_import: bool = False, import_location=None):
        """Set or sample the next navigation goal."""
        super().generate_environment()

        if self.external_goal is not None:
            self.goal_location = self.external_goal.copy()
        elif manual_import is False:
            self.goal_location = self.sample_goal(self.auv_attitude[2])
        else:
            self.goal_location = np.asarray(import_location, dtype=np.float32)

    def _update_navigation_state(self) -> dict[str, float]:
        """Refresh navigation errors with the legacy Task1 pitch convention."""
        self.episode_state.navigation.position = self.auv_position.copy()
        self.episode_state.navigation.attitude_rpy = self.auv_attitude.copy()
        self.episode_state.navigation.relative_velocity = self.auv_relative_velocity.copy()
        self.episode_state.navigation.angular_velocity = self.auv_angular_velocity.copy()
        self.episode_state.navigation.goal_location = (
            None if self.goal_location is None else np.asarray(self.goal_location, dtype=np.float32)
        )
        self.episode_state.navigation.goal_heading = self.heading_goal_reached
        self.episode_state.navigation.last_attitude_rpy = self.auv_last_attitude.copy()
        self.episode_state.navigation.last_position = self.auv_last_position.copy()

        diff = self.goal_location - self.auv_position
        self.delta_d = float(np.linalg.norm(diff))
        self.delta_theta = float(
            self.auv_attitude[1] + self.ssa(np.arctan2(diff[2], np.linalg.norm(diff[:2])))
        )
        self.delta_psi = float(self.ssa(np.arctan2(diff[1], diff[0]) - self.auv_attitude[2]))
        self.delta_heading_goal = float(self.ssa(self.heading_goal_reached - self.auv_attitude[2]))
        self.stable_delta_theta = float(self.auv_attitude[1] - self.auv_last_attitude[1])
        self.distance_moved_per_step = float(np.linalg.norm(self.auv_position - self.auv_last_position))

        return {
            "delta_d": self.delta_d,
            "delta_theta": self.delta_theta,
            "delta_psi": self.delta_psi,
            "delta_heading_goal": self.delta_heading_goal,
            "stable_delta_theta": self.stable_delta_theta,
            "distance_moved_per_step": self.distance_moved_per_step,
        }

#判定碰撞
    def update_body_collision(self, radar_dist: np.ndarray) -> bool:
        """Detect collision from raw Task1 rays before reduced observation pooling."""
        raw_distances = getattr(self.rangefinder, "intersection_distances", None)
        if raw_distances is not None and len(raw_distances) > 0:
            return bool(np.any(np.asarray(raw_distances) == 0))
        return super().update_body_collision(radar_dist)

    def reward_step_impro(self, action: np.ndarray) -> float:
        """Compute the Task1 reward through the skill reward module."""
        return compute_task1_reward(self, action)

    def is_done(self):

        # 避障恢复
        recovering = getattr(self, "collision_recovery_active", False)
        recovery_failed = getattr(self, "collision_recovery_failed", False)
        success = bool(
            (not recovering)
            and (float(self.delta_d) < float(self.dist_goal_reached_tol))
            # To restore the stricter Env_Task1 terminal check, also require:
            # and np.linalg.norm(self.auv_relative_velocity) < self.velocity_goal_reached_tol
            # np.linalg.norm(self.auv_angular_velocity) < self.ang_rate_goal_reached_tol
            # and abs(self.auv_attitude[0]) < self.attitude_goal_reached_tol
            # and abs(self.auv_attitude[1]) < self.attitude_goal_reached_tol
        )

        out_of_dist = bool(float(self.delta_d) > float(self.max_dist_from_goal))

        out_of_att = bool(
            abs(float(self.auv_attitude[0])) > float(self.max_attitude)
            or abs(float(self.auv_attitude[1])) > float(self.max_attitude)
        )
        timeout = bool(int(self.t_steps) >= int(self.max_timesteps))

        self.conditions = [success, out_of_dist, out_of_att, timeout, recovery_failed]
        # self.conditions = [success, out_of_dist, out_of_att, timeout, self.collision]
        done = bool(any(self.conditions))

        self.done = done
        self.goal_reached = bool(success)
        self.episode_state.done = done
        self.episode_state.goal_reached = self.goal_reached

        # 注意：这里保留真实碰撞状态，但不让普通碰撞直接 done
        self.episode_state.collision = bool(self.collision)

        cond_idx = [index for index, flag in enumerate(self.conditions) if flag]
        if done:
            print(
                "navigation done | "
                f"goal_pos={np.asarray(self.goal_location).round(3).tolist()} | "
                f"cur_pos={np.asarray(self.auv_position).round(3).tolist()} | "
                f"delta_d={self.delta_d:.3f} | "
                f"conditions={cond_idx}"
            )
        return done, cond_idx
