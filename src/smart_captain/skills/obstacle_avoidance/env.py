"""Obstacle-avoidance skill environment."""

from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import numpy as np
from skimage.measure import block_reduce

from smart_captain.simulation.compat import BASE_CONFIG, BaseEnvironment, Reward
from smart_captain.skills.base import SkillAdapter
from smart_captain.skills.obstacle_avoidance.config import (
    OBSTACLE_AVOIDANCE_SPEC,
    ObstacleAvoidanceConfig,
)


class ObstacleAvoidanceEnv(BaseEnvironment, SkillAdapter):
    """Native obstacle-avoidance environment path for the new framework."""

    spec = OBSTACLE_AVOIDANCE_SPEC

    def __init__(self, env_config: dict = BASE_CONFIG, auv=None, train_mode: bool = True):
        self.skill_config = ObstacleAvoidanceConfig(
            observation_dim=40,
            reduced_horizontal_indices=(20, 21, 22, 23, 0, 1, 2, 3, 4),
            vertical_reduce_stride=3,
            r_min=env_config.get("r_min", 30.0),
            r_max=env_config.get("r_max", 40.0),
            z_min=-15.0,
            z_max=-5.0,
        )
        SkillAdapter.__init__(self, runtime_env=auv)
        BaseEnvironment.__init__(self, env_config, auv, train_mode)
        self.r_min = self.skill_config.r_min
        self.r_max = self.skill_config.r_max

        obs_low = -np.ones(self.skill_config.observation_dim)
        obs_low[0] = 0
        obs_low[13:] = 0
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=np.ones(self.skill_config.observation_dim),
            dtype=np.float32,
        )

        self.vertical_angles = np.array([-60, -45, -30, -15, 0, 15, 30, 45, 60]) * np.pi / 180.0
        self.radar_angles_deg = [0, 15, 30, 45, 60, 300, 315, 330, 345]
        self.horizontal_angles = np.deg2rad(self.radar_angles_deg)
        self.horizontal_angles = (self.horizontal_angles + np.pi) % (2 * np.pi) - np.pi

        self.n_vertical = len(self.vertical_angles)
        self.n_horizontal = len(self.horizontal_angles)
        self.n_rays = self.n_vertical * self.n_horizontal

        self.alpha = np.repeat(self.vertical_angles, self.n_horizontal)
        self.alpha_max = np.pi / 3
        self.beta = np.tile(self.horizontal_angles, self.n_vertical)
        self.beta_max = np.pi / 3

    def sample_goal(self, auv_yaw: float) -> np.ndarray:
        """Sample a navigation target using the migrated task2 logic."""
        r_target = np.random.uniform(self.skill_config.r_min, self.skill_config.r_max)
        psi_target_rel = np.random.uniform(-np.pi, np.pi)
        psi_target_abs = self.ssa(auv_yaw + psi_target_rel)
        x_target = r_target * np.cos(psi_target_abs)
        y_target = r_target * np.sin(psi_target_abs)
        z_target = np.random.uniform(self.skill_config.z_min, self.skill_config.z_max)
        return np.array([x_target, y_target, z_target], dtype=np.float32)

    def generate_environment(self, manual_import: bool = False, import_location=None):
        """Set or sample the next obstacle-avoidance goal."""
        super().generate_environment()
        if manual_import is False:
            self.goal_location = self.sample_goal(self.auv_attitude[2])
        else:
            self.goal_location = np.asarray(import_location, dtype=np.float32)

    def reset(self, manual_import: bool = False, import_location=None):
        _, return_info_dict = super().reset()
        observation = np.zeros(self.skill_config.observation_dim, dtype=np.float32)
        self.generate_environment(manual_import, import_location)
        self.update_navigation_errors()
        self.delta_d_list.append(self.delta_d)
        return observation, return_info_dict

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, _, info = super().step(action)
        self.intersec_dist = self.intersec_dist_reduced()
        self.obser = self.task_observe(obs)
        self.reward = self.reward_step(reward)
        self.done = done
        self.info = info
        return self.obser, self.reward, self.done, False, self.info

    def intersec_dist_reduced(self):
        raw_dist = self.radar.intersection_distances
        n_vert = self.radar.n_vertical
        n_horiz = self.radar.n_horizontal
        dist_mat = raw_dist.reshape((n_vert, n_horiz))
        sub_mat = dist_mat[:, list(self.skill_config.reduced_horizontal_indices)]
        self.dist = sub_mat.flatten()
        reduced = block_reduce(
            sub_mat,
            block_size=(self.skill_config.vertical_reduce_stride, 1),
            func=np.min,
        )
        return reduced.flatten()

    def task_observe(self, obs):
        obser = np.zeros(self.skill_config.observation_dim, dtype=np.float32)
        obser[0:13] = obs[0:13]
        obser[13:] = np.clip(self.intersec_dist / self.radar_max_dist, 0, 1)
        return obser

    def reward_step(self, reward):
        reward = reward - self.last_reward_arr[6]
        self.last_reward_arr[6] = -self.reward_factors["w_oa"] * Reward.obstacle_avoidance_improve(
            theta_r=self.alpha,
            psi_r=self.beta,
            d_r=self.dist,
            theta_max=self.alpha_max,
            psi_max=self.beta_max,
            d_max=self.radar.max_dist,
            gamma_c=1,
            epsilon_c=0.001,
            epsilon_oa=0.01,
        )
        reward = reward + self.last_reward_arr[6]
        return reward
