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
from smart_captain.simulation.defaults import DEFAULT_ENV_CONFIG

class BaseEnvironment(BaseSimulationAdapter):
    """Migration-phase base environment implemented inside the new framework."""

    def __init__(self, config: dict = DEFAULT_ENV_CONFIG, auv=None, train_mode: bool = True):
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
        self.max_timesteps = self.config["max_timesteps"]
        self.interval_datastorage = self.config["interval_datastorage"]
        self.heading_goal_reached = 0.0
        self.goal_location = None
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
        self.t_steps = self.episode_state.t_steps
        self.t_total_steps = self.episode_state.t_total_steps
        self.episode = self.episode_state.episode
        self.last_reward = self.episode_state.last_reward
        self.cumulative_reward = self.episode_state.cumulative_reward

        obs_low = -np.ones(self.n_observations, dtype=np.float32)
        obs_low[0] = 0
        obs_low[13:] = 0
        self.action_space = gym.spaces.Box(
            low=np.array([-40, -20, -10, -20], dtype=np.float32),
            high=np.array([40, 20, 10, 20], dtype=np.float32),
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

        return holoocean.make(scenario_cfg=self.scenario_cfg, show_viewport=True)

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
        self.delta_d_list.append(errors["delta_d"])
        if return_info:
            return self.observation, return_info_dict
        return self.observation

    def generate_environment(self):
        """Default goal generator used by migrated legacy tasks."""
        self.goal_location = np.array([30, 0, 0], dtype=np.float32)
        self.heading_goal_reached = 0.0

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
        """Evaluate terminal conditions."""
        errors = self._update_navigation_state()
        self.episode_state.collision = self.collision
        done, cond_idx, conditions = self.done_evaluator.evaluate(self.episode_state, errors)
        self.conditions = conditions
        self.done = done
        self.goal_reached = self.episode_state.goal_reached
        return done, cond_idx

    def reward_step_impro(self, action: np.ndarray, dt: float = 0.005):
        """Compute the migrated legacy reward."""
        if len(self.delta_d_list) < 2:
            self.delta_d_list.extend([self.delta_d, self.delta_d])
        delta_d = self.delta_d_list[-1] - self.delta_d_list[-2]
        self.last_reward_arr[0] = -delta_d / dt

        self.last_reward_arr[1] = -self.reward_factors["w_delta_theta"] * RewardMath.cont_goal_constraints(
            x=np.abs(self.delta_theta),
            delta_d=self.delta_d,
            x_des=0.0,
            delta_d_des=self.dist_goal_reached_tol,
            x_max=np.pi / 2,
            delta_d_max=self.max_dist_from_goal,
            x_exp=4.0,
            delta_d_exp=0.0,
            x_rev=False,
            delta_d_rev=False,
        )

        self.last_reward_arr[2] = self.reward_factors["w_delta_psi"] * RewardMath.angle_control(
            r=np.abs(self.delta_psi),
            r_goal=0.0,
            r_max=np.pi,
            r_partition=np.pi / 6,
            r_exp=4,
        )

        self.last_reward_arr[3] = -self.reward_factors["w_phi"] * (self.auv_attitude[0] / (np.pi / 2)) ** 2
        self.last_reward_arr[4] = 0.0
        self.last_reward_arr[5] = 0.0
        self.last_reward_arr[6] = -self.reward_factors["w_oa"] * RewardMath.obstacle_avoidance_improve(
            theta_r=self.radar.alpha,
            psi_r=self.radar.beta,
            d_r=self.radar.intersection_distances,
            theta_max=self.radar.alpha_max,
            psi_max=self.radar.beta_max,
            d_max=self.radar.max_dist,
            gamma_c=1,
            epsilon_c=0.001,
            epsilon_oa=0.01,
        )
        self.last_reward_arr[7] = 0.0
        self.last_reward_arr[self.episode_state.continuous_reward_dim :] = np.array(self.conditions) * self.w_done

        self.cum_reward_arr = self.cum_reward_arr + self.last_reward_arr
        reward = float(np.sum(self.last_reward_arr))
        self.last_reward = reward
        self.episode_state.last_reward = reward
        return reward

    def step(self, action: np.ndarray):
        """Step the environment one control action."""
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

        self.collision = self.update_body_collision(self.radar_intersec_dist)
        self.episode_state.collision = self.collision
        errors = self._update_navigation_state()
        self.delta_d_list.append(errors["delta_d"])
        self.observation = self.observe()
        self.done, cond_idx = self.is_done()
        self.last_reward = self.reward_step_impro(action)

        self.episode_state.cumulative_reward += self.last_reward
        self.episode_state.t_total_steps += 1
        self.episode_state.t_steps += 1
        self.auv_last_attitude = self.auv_attitude.copy()
        self.auv_last_position = self.auv_position.copy()
        self.last_radar_r = self.radar_intersec_dist.copy()
        self._sync_episode_mirrors()

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
            "position": self.auv_position,
        }
        return self.observation, self.last_reward, self.done, False, self.info

    def sync_state_from(self, other):
        """Copy transition-critical state from another env instance."""
        if hasattr(other, "episode_state"):
            self.episode_state.sync_from(other.episode_state)
            self._sync_episode_mirrors()
        self.auv_last_attitude = other.auv_last_attitude.copy()
        self.auv_last_position = other.auv_last_position.copy()
        self.last_radar_r = other.last_radar_r.copy() if hasattr(other, "last_radar_r") else self.last_radar_r
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
