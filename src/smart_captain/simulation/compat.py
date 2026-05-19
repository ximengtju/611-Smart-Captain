"""Local compatibility layer for migrated simulation code."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from smart_captain.simulation.core.base_env import (
    BaseSimulationAdapter,
    NavigationState,
    ObservationNormalizationConfig,
    RewardMath,
    RuntimeEpisodeState,
)
from smart_captain.simulation.core.action_mapper import CANONICAL_ACTION_HIGH, CANONICAL_ACTION_LOW
from smart_captain.simulation.defaults import DEFAULT_ENV_CONFIG

class BaseEnvironment(BaseSimulationAdapter, gym.Env):
    """Migration-phase base environment implemented inside the new framework."""

    def __init__(self, config: dict = DEFAULT_ENV_CONFIG, auv=None, train_mode: bool = True):
        gym.Env.__init__(self)
        normalization_config = ObservationNormalizationConfig(
            radar_max_dist=config["radar.max_dist"],
            u_max=config["u_max"],
            v_max=config["v_max"],
            w_max=config["w_max"],
            p_max=config["p_max"],
            q_max=config["q_max"],
            r_max=config["r_max"],
            max_attitude=config["max_attitude"],
            dist_goal_reached_tol=config["dist_goal_reached_tol"],
            max_dist_from_goal=config["max_dist_from_goal"],
            max_timesteps=config["max_timesteps"],
        )
        super().__init__(normalization_config)
        self.config = config
        self.scenario_cfg = self.config["auv_config"]
        self.dt = 1.0 / float(self.scenario_cfg.get("ticks_per_sec", 200))
        self.AUV = auv if not train_mode else self._make_runtime_auv()

        self.log_level = self.config["log_level"]
        self.save_path_folder = self.config["save_path_folder"]
        self.title = self.config["title"]
        self.verbose = self.config["verbose"]
        self.interval_episode_log = self.config["interval_episode_log"]

        self.n_observations = self.config["n_observations"]
        self.n_actions = self.config["n_actions"]
        self.radar_max_dist = self.config["radar.max_dist"]
        self.observation = np.zeros(self.n_observations, dtype=np.float32)
        self.info: dict[str, Any] = {}
        self.conditions = None

        self.reward_factors = self.config["reward_factors"]
        self.w_done = np.array(
            [
                self.reward_factors["w_goal"],
                self.reward_factors["w_deltad_max"],
                self.reward_factors["w_Theta_max"],
                self.reward_factors["w_t_max"],
                self.reward_factors["w_col"],
            ],
            dtype=np.float32,
        )

        self.meta_data_done = [
            "Done-Goal_reached",
            "Done-out_pos",
            "Done-out_att",
            "Done-max_t",
            "Done-collision",
        ]

        self.action_reward_factors = self.config["action_reward_factors"]
        self.dist_goal_reached_tol = self.config["dist_goal_reached_tol"]
        self.velocity_goal_reached_tol = self.config["velocity_goal_reached_tol"]
        self.ang_rate_goal_reached_tol = self.config["ang_rate_goal_reached_tol"]
        self.attitude_goal_reached_tol = self.config["attitude_goal_reached_tol"]
        self.max_dist_from_goal = self.config["max_dist_from_goal"]
        self.max_attitude = self.config["max_attitude"]
        self.z_goal_tol = self.config.get("z_goal_reached_tol", 2.5)
        self.depth_soft_tol = self.config.get("depth_soft_tol", 6.0)
        self.depth_gate_dist = self.config.get("depth_gate_dist", 40.0)
        self.depth_band_margin = self.config.get("depth_band_margin", 8.0)
        self.max_timesteps = self.config["max_timesteps"]
        self.interval_datastorage = self.config["interval_datastorage"]
        self.heading_goal_reached = 0.0
        self.goal_location = None
        self.external_goal = None
        self.goal_constraints: list[Any] = []

        self.auv_attitude = np.zeros(3, dtype=np.float32)
        self.auv_position = np.zeros(3, dtype=np.float32)
        self.auv_relative_velocity = np.zeros(3, dtype=np.float32)
        self.auv_angular_velocity = np.zeros(3, dtype=np.float32)
        self.auv_last_attitude = np.zeros(3, dtype=np.float32)
        self.auv_last_position = np.zeros(3, dtype=np.float32)
        self.radar_intersec_dist = np.zeros(24, dtype=np.float32)
        self.last_radar_r = np.zeros(24, dtype=np.float32)
        self.delta_d = 0.0
        self.delta_theta = 0.0
        self.delta_psi = 0.0
        self.delta_heading_goal = 0.0
        self.delta_d_xy_last = 0.0
        self.delta_d_init = 0.0
        self.z_start = 0.0
        self.stable_delta_theta = 0.0
        self.distance_moved_per_step = 0.0

        self.episode_state = RuntimeEpisodeState(
            navigation=NavigationState(
                position=self.auv_position.copy(),
                attitude_rpy=self.auv_attitude.copy(),
                relative_velocity=self.auv_relative_velocity.copy(),
                angular_velocity=self.auv_angular_velocity.copy(),
                goal_location=None,
                goal_heading=self.heading_goal_reached,
                last_attitude_rpy=self.auv_last_attitude.copy(),
                last_position=self.auv_last_position.copy(),
            ),
            observation_dim=self.n_observations,
        )
        self.delta_d_list = self.episode_state.delta_d_history
        self.last_reward_arr = self.episode_state.last_reward_arr
        self.cum_reward_arr = self.episode_state.cumulative_reward_arr

        self.collision = self.episode_state.collision
        self.goal_reached = self.episode_state.goal_reached
        self.done = self.episode_state.done
#######################################################################################
        #碰撞恢复
        self.collision_distance = float(self.config.get("collision_distance", 0.7))
        self.collision_recovery_safe_distance = float(self.config.get("collision_recovery_safe_distance", 8.0))
        self.collision_recovery_stop_steps = int(self.config.get("collision_recovery_stop_steps", 400))
        self.collision_recovery_clear_steps = int(self.config.get("collision_recovery_clear_steps", 100))
        self.collision_recovery_max_steps = int(self.config.get("collision_recovery_max_steps", 2000))
        self.collision_recovery_heave = float(self.config.get("collision_recovery_heave", 8.0))

        self.nearest_obstacle_distance = None
        self.collision_recovery_active = False
        self.collision_recovery_step = 0
        self.collision_recovery_clear_count = 0
        self.collision_recovery_failed = False
        self.collision_count = 0
#######################################################################################
        self.t_steps = self.episode_state.t_steps
        self.t_total_steps = self.episode_state.t_total_steps
        self.episode = self.episode_state.episode
        self.last_reward = self.episode_state.last_reward
        self.cumulative_reward = self.episode_state.cumulative_reward

        obs_low = -np.ones(self.n_observations, dtype=np.float32)
        obs_low[0] = 0
        obs_low[13:] = 0
        self.action_space = gym.spaces.Box(
            low=CANONICAL_ACTION_LOW.copy(),
            high=CANONICAL_ACTION_HIGH.copy(),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=np.ones(self.n_observations, dtype=np.float32),
            dtype=np.float32,
        )

    def _make_runtime_auv(self):
        """Create the HoloOcean runtime only when actually needed."""
        import holoocean

        return holoocean.make(
            scenario_cfg=self.scenario_cfg,
            show_viewport=self.config.get("show_viewport", False),
        )

    def _sync_episode_mirrors(self) -> None:
        """Keep legacy attribute names aligned with the episode-state container."""
        self.delta_d_list = self.episode_state.delta_d_history
        self.last_reward_arr = self.episode_state.last_reward_arr
        self.cum_reward_arr = self.episode_state.cumulative_reward_arr
        self.collision = self.episode_state.collision
        self.goal_reached = self.episode_state.goal_reached
        self.done = self.episode_state.done
        self.t_steps = self.episode_state.t_steps
        self.t_total_steps = self.episode_state.t_total_steps
        self.episode = self.episode_state.episode
        self.last_reward = self.episode_state.last_reward
        self.cumulative_reward = self.episode_state.cumulative_reward

    def _update_navigation_state(self) -> dict[str, float]:
        """Refresh the container state from legacy attribute fields."""
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
        errors = self.episode_state.navigation.update_navigation_errors()
        self.delta_d = errors["delta_d"]
        self.delta_theta = errors["delta_theta"]
        self.delta_psi = errors["delta_psi"]
        self.delta_heading_goal = errors["delta_heading_goal"]
        self.stable_delta_theta = errors["stable_delta_theta"]
        self.distance_moved_per_step = errors["distance_moved_per_step"]
        return errors

    def reset(self, seed=None, return_info: bool = True, options=None):
        """Reset the local environment runtime."""
        return_info_dict = self.info.copy()
        sensor_return = self.AUV.reset()
        dynamics_obs = sensor_return["DynamicsSensor"]
        velocity_obs = sensor_return["VelocitySensor"]

        self.auv_last_attitude = np.zeros(3, dtype=np.float32)
        self.auv_last_position = np.zeros(3, dtype=np.float32)
        self.auv_attitude = self.angle_conversion(dynamics_obs[15:18])
        self.auv_position = dynamics_obs[6:9].astype(np.float32)
        self.auv_last_attitude = self.auv_attitude.copy()
        self.auv_last_position = self.auv_position.copy()
        self.auv_relative_velocity = np.asarray(velocity_obs, dtype=np.float32)
        self.auv_angular_velocity = dynamics_obs[12:15].astype(np.float32)
        self.rangefinder.update_from_sensor_return(sensor_return)
        self.radar = self.rangefinder
        self.radar_intersec_dist = self.rangefinder.reduced_distances

        self.info = {}
        self.goal_constraints = []
        self.goal_reached = False
        self.collision = False
        self.observation = np.zeros(self.n_observations, dtype=np.float32)
        # 碰撞恢复
        # 每次新 episode 开始时，必须把上一轮碰撞恢复状态清掉。否则上一轮如果还在恢复或恢复失败，下一轮可能会继承错误状态。
        self.nearest_obstacle_distance = None
        self.collision_recovery_active = False
        self.collision_recovery_step = 0
        self.collision_recovery_clear_count = 0
        self.collision_recovery_failed = False
        self.collision_count = 0

        self.episode_state = RuntimeEpisodeState(
            navigation=NavigationState(
                position=self.auv_position.copy(),
                attitude_rpy=self.auv_attitude.copy(),
                relative_velocity=self.auv_relative_velocity.copy(),
                angular_velocity=self.auv_angular_velocity.copy(),
                goal_location=None,
                goal_heading=self.heading_goal_reached,
                last_attitude_rpy=self.auv_last_attitude.copy(),
                last_position=self.auv_last_position.copy(),
            ),
            observation_dim=self.n_observations,
        )
        self._sync_episode_mirrors()
        self.episode_state.episode += 1
        self._sync_episode_mirrors()

        if seed is not None:
            np.random.seed(seed)

        self.generate_environment()
        errors = self._update_navigation_state()
        diff = self.goal_location - self.auv_position
        self.delta_d_xy_last = float(np.linalg.norm(diff[:2]))
        self.delta_d_init = float(errors["delta_d"])
        self.z_start = float(self.auv_position[2])
        self.delta_d_list.append(errors["delta_d"])
        self.observation = self.observe()
        if return_info:
            return self.observation, return_info_dict
        return self.observation

    def generate_environment(self):
        """Default goal generator used by migrated legacy tasks."""
        if self.external_goal is not None:
            self.goal_location = self.external_goal.copy()
        else:
            self.goal_location = np.array([30, 0, 0], dtype=np.float32)
        self.heading_goal_reached = 0.0

    def set_goal(self, goal) -> None:
        """Set an externally controlled goal for evaluation and demos."""
        self.external_goal = np.asarray(goal, dtype=np.float32)

    def clear_goal(self) -> None:
        """Clear an externally controlled goal and return to sampled goals."""
        self.external_goal = None

    def update_navigation_errors(self, location=None):
        """Legacy-compatible navigation error update entrypoint."""
        if location is not None:
            self.goal_location = np.asarray(location, dtype=np.float32)
        return self._update_navigation_state()

    @staticmethod
    def ssa(angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def observe(self):
        """Build the normalized observation vector."""
        errors = self._update_navigation_state()
        return self.observation_builder.build(self.episode_state, errors, self.radar_intersec_dist)

    def is_done(self):
        """Evaluate Task4/OpenWater terminal conditions."""
        errors = self._update_navigation_state()
        z_err = abs(self.goal_location[2] - self.auv_position[2])
        delta_d_xy = np.linalg.norm((self.goal_location - self.auv_position)[:2])
        success = delta_d_xy < self.dist_goal_reached_tol and z_err < self.z_goal_tol
        out_of_dist = errors["delta_d"] > self.max_dist_from_goal
        out_of_att = (
            abs(self.auv_attitude[0]) > self.max_attitude
            or abs(self.auv_attitude[1]) > self.max_attitude
        )
        timeout = self.t_steps >= self.max_timesteps

        # terminated = bool(success or out_of_dist or out_of_att or self.collision)
        # truncated = bool(timeout and not terminated)
        # conditions = [success, out_of_dist, out_of_att, truncated, self.collision]
        # 碰撞恢复新版，原来的 collision -> terminal_collision ，恢复失败：才真正结束 episode
        if self.collision_recovery_active:
            success = False

        terminal_collision = self.collision_recovery_failed

        terminated = bool(success or out_of_dist or out_of_att or terminal_collision)
        truncated = bool(timeout and not terminated)
        conditions = [success, out_of_dist, out_of_att, truncated, terminal_collision]

        self.episode_state.collision = self.collision
        done = terminated or truncated
        cond_idx = [index for index, flag in enumerate(conditions) if flag]
        self.conditions = conditions
        self.done = done
        self.goal_reached = bool(success)
        self.episode_state.goal_reached = self.goal_reached
        return done, cond_idx

    def reward_step_impro(self, action: np.ndarray):
        """Compute the migrated Task4/OpenWater reward."""
        self.last_reward_arr[:] = 0.0
        if len(self.delta_d_list) < 2:
            self.delta_d_list.extend([self.delta_d, self.delta_d])
        delta_d = self.delta_d_list[-1] - self.delta_d_list[-2]
        self.last_reward_arr[0] = -delta_d / max(self.dt, 1e-6)

        self.last_reward_arr[1] = -self.reward_factors["w_delta_theta"] * RewardMath.cont_goal_constraints(
            x=np.abs(self.delta_theta),
            delta_d=self.delta_d,
            x_des=0.0,
            delta_d_des=self.dist_goal_reached_tol,
            x_max=np.pi / 2,
            delta_d_max=self.max_dist_from_goal,
            x_exp=2.0,
            delta_d_exp=0.0,
            x_rev=False,
            delta_d_rev=False,
        )

        self.last_reward_arr[2] = self.reward_factors["w_delta_psi"] * RewardMath.angle_control(
            r=np.abs(self.delta_psi),
            r_goal=0.0,
            r_max=np.pi,
            r_partition=np.pi / 6,
            r_exp=2,
        )

        self.last_reward_arr[3] = -self.reward_factors["w_phi"] * (self.auv_attitude[0] / (np.pi / 2)) ** 2
        self.last_reward_arr[4] = -self.reward_factors["w_theta"] * (
            (self.auv_attitude[1] - self.auv_last_attitude[1]) / (np.pi / 2)
        ) ** 2
        self.last_reward_arr[5] = 0.0

        beta = self.radar.beta_reduced
        d = self.radar_intersec_dist
        goal_mask = np.abs(self.ssa(beta - self.delta_psi)) < np.deg2rad(30)
        d_goal = float(np.min(d[goal_mask])) if np.any(goal_mask) else float(self.radar_max_dist)
        d_min = float(np.min(d))
        r_block_goal = -2.0 * max(0.0, (5.0 - d_goal) / 5.0) ** 2
        r_too_close = -3.0 * max(0.0, (3.0 - d_min) / 3.0) ** 2
        speed = float(np.linalg.norm(self.auv_relative_velocity))
        stuck = d_min < 2.0 and speed < 0.2
        r_stuck = -1.0 if stuck else 0.0
        self.last_reward_arr[6] = self.reward_factors["w_oa"] * (r_block_goal + r_too_close + r_stuck)
        self.last_reward_arr[7] = 0.0
        self.last_reward_arr[self.episode_state.continuous_reward_dim :] = np.array(self.conditions) * self.w_done

        if self.goal_reached:
            remaining_steps = self.max_timesteps - self.t_steps
            self.last_reward_arr[8] += 0.05 * remaining_steps

        self.cum_reward_arr = self.cum_reward_arr + self.last_reward_arr
        reward = float(np.sum(self.last_reward_arr))
        self.last_reward = reward
        self.episode_state.last_reward = reward
        self.delta_d_xy_last = float(np.linalg.norm((self.goal_location - self.auv_position)[:2]))
        return reward

#######################################################################
#碰撞增加恢复辅助代码
    def _nearest_range_distance(self) -> float | None:
        raw_distances = getattr(self.rangefinder, "intersection_distances", None)
        distances = raw_distances if raw_distances is not None and len(raw_distances) > 0 else self.radar_intersec_dist

        try:
            distances = np.asarray(distances, dtype=np.float32).reshape(-1)
        except Exception:
            return None

        finite_distances = distances[np.isfinite(distances)]
        if finite_distances.size == 0:
            return None
        return float(np.min(finite_distances))

    #启动碰撞恢复流程
    def _activate_collision_recovery(self) -> None:
        if not self.collision_recovery_active:
            self.collision_count += 1
            self.collision_recovery_step = 0
            self.collision_recovery_clear_count = 0
            self.collision_recovery_failed = False
        self.collision_recovery_active = True

    # 碰撞恢复期间的动作
    def _collision_recovery_action(self) -> np.ndarray:
        if self.collision_recovery_step < self.collision_recovery_stop_steps:
            return np.zeros(self.n_actions, dtype=np.float32)

        return np.array(
            [0.0, 0.0, self.collision_recovery_heave, 0.0],
            dtype=np.float32,
        )

    #每一步更新恢复状态，判断恢复是否成功或失败。
    def _update_collision_recovery(self) -> None:
        if not self.collision_recovery_active:
            return

        self.collision_recovery_step += 1

        if (
                self.nearest_obstacle_distance is not None
                and self.nearest_obstacle_distance > self.collision_recovery_safe_distance
        ):
            self.collision_recovery_clear_count += 1
        else:
            self.collision_recovery_clear_count = 0

        if self.collision_recovery_clear_count >= self.collision_recovery_clear_steps:
            self.collision_recovery_active = False
            self.collision_recovery_step = 0
            self.collision_recovery_clear_count = 0
            self.collision = False
            self.episode_state.collision = False
            return

        if self.collision_recovery_step >= self.collision_recovery_max_steps:
            self.collision_recovery_active = False
            self.collision_recovery_failed = True

#########################################################################

    def step(self, action: np.ndarray):

        # action = np.asarray(action, dtype=np.float32)
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        # command = self.action_to_command(action)

        ##增加碰撞恢复动作考虑
        # 未碰撞：使用模型动作
        # 碰撞恢复中：忽略模型动作，强制执行恢复动作
        if self.collision_recovery_active:
            action = self._collision_recovery_action()
        else:
            action = np.asarray(action, dtype=np.float32)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        command = self.action_to_command(action)

        sensor_return = self.AUV.step(command)
        dynamics_obs = sensor_return["DynamicsSensor"]
        velocity_obs = sensor_return["VelocitySensor"]
        self.auv_attitude = self.angle_conversion(dynamics_obs[15:18])
        self.auv_position = dynamics_obs[6:9].astype(np.float32)
        self.auv_relative_velocity = np.asarray(velocity_obs, dtype=np.float32)
        self.auv_angular_velocity = dynamics_obs[12:15].astype(np.float32)
        self.rangefinder.update_from_sensor_return(sensor_return)
        self.radar = self.rangefinder
        self.radar_intersec_dist = self.rangefinder.reduced_distances

        # self.collision = self.update_body_collision(self.radar_intersec_dist)
        # self.episode_state.collision = self.collision
        # 新版碰撞检测
        self.nearest_obstacle_distance = self._nearest_range_distance()
        self.collision = (
                self.nearest_obstacle_distance is not None
                and self.nearest_obstacle_distance <= self.collision_distance
        )

        if self.collision:
            self._activate_collision_recovery()

        self._update_collision_recovery()
        self.episode_state.collision = self.collision

        errors = self._update_navigation_state()
        self.delta_d_list.append(errors["delta_d"])
        if len(self.delta_d_list) > 2:
            self.delta_d_list = self.delta_d_list[-2:]
            self.episode_state.delta_d_history = self.delta_d_list
        self.observation = self.observe()

        self.episode_state.t_total_steps += 1
        self.episode_state.t_steps += 1
        self._sync_episode_mirrors()
        self.done, cond_idx = self.is_done()
        self.last_reward = self.reward_step_impro(action)
        self.episode_state.cumulative_reward += self.last_reward
        self.auv_last_attitude = self.auv_attitude.copy()
        self.auv_last_position = self.auv_position.copy()
        self.last_radar_r = self.radar_intersec_dist.copy()
        self._sync_episode_mirrors()

        # info新增 碰撞恢复
        self.info = {
            "episode_number": self.episode,
            "t_step": self.t_steps,
            "t_total_steps": self.t_total_steps,
            "cumulative_reward": self.cumulative_reward,
            "last_reward": self.last_reward,
            "done": self.done,
            "conditions_true": cond_idx,
            "conditions_true_info": [self.meta_data_done[i] for i in cond_idx],
            "collision": self.collision,
            "goal_reached": self.goal_reached,
            "delta_d": self.delta_d,
            "delta_d_xy": float(np.linalg.norm((self.goal_location - self.auv_position)[:2])),
            "z_err": float(abs(self.goal_location[2] - self.auv_position[2])),
            "w_abs": float(abs(self.auv_relative_velocity[2])),
            "position": self.auv_position.copy(),

            "nearest_obstacle_distance": self.nearest_obstacle_distance,
            "collision_recovery_active": self.collision_recovery_active,
            "collision_recovery_step": self.collision_recovery_step,
            "collision_recovery_clear_count": self.collision_recovery_clear_count,
            "collision_recovery_failed": self.collision_recovery_failed,
            "collision_count": self.collision_count,
            "applied_action": action.copy(),
        }
        return self.observation, self.last_reward, self.done, False, self.info

    # def sync_state_from(self, other):
    #     """Copy transition-critical state from another env instance."""
    #     if hasattr(other, "episode_state"):
    #         self.episode_state.sync_from(other.episode_state)
    #         self._sync_episode_mirrors()
    #     self.auv_last_attitude = other.auv_last_attitude.copy()
    #     self.auv_last_position = other.auv_last_position.copy()
    #     self.last_radar_r = other.last_radar_r.copy() if hasattr(other, "last_radar_r") else self.last_radar_r
    #     self.goal_location = other.goal_location.copy() if other.goal_location is not None else None
    #     self.heading_goal_reached = other.heading_goal_reached
    #
    def sync_state_from(self, other):
        """Copy transition-critical state from another env instance."""
        if hasattr(other, "episode_state"):
            self.episode_state.sync_from(other.episode_state)
            self._sync_episode_mirrors()

        self.auv_attitude = other.auv_attitude.copy()
        self.auv_position = other.auv_position.copy()
        self.auv_relative_velocity = other.auv_relative_velocity.copy()
        self.auv_angular_velocity = other.auv_angular_velocity.copy()

        self.auv_last_attitude = other.auv_last_attitude.copy()
        self.auv_last_position = other.auv_last_position.copy()

        self.radar_intersec_dist = other.radar_intersec_dist.copy()
        self.last_radar_r = (
            other.last_radar_r.copy()
            if hasattr(other, "last_radar_r")
            else self.radar_intersec_dist.copy()
        )

        if hasattr(other, "rangefinder") and hasattr(self, "rangefinder"):
            self.rangefinder.intersection_distances = other.rangefinder.intersection_distances.copy()
            self.radar = self.rangefinder

        self.collision = other.collision
        self.episode_state.collision = self.collision

        self.goal_location = other.goal_location.copy() if other.goal_location is not None else None
        self.heading_goal_reached = other.heading_goal_reached

def __getattr__(name: str):
    """Resolve compatibility symbols lazily."""
    if name == "BASE_CONFIG":
        return DEFAULT_ENV_CONFIG
    if name == "Reward":
        return RewardMath
    if name == "BaseEnvironment":
        return BaseEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BASE_CONFIG", "BaseEnvironment", "Reward"]
