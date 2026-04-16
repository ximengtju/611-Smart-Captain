"""Shared environment primitives for the restructured simulation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from smart_captain.simulation.core.action_mapper import DEFAULT_ACTION_MAPPER
from smart_captain.simulation.core.sensor_processor import RangeFinderGrid, wrap_to_pi


@dataclass(frozen=True)
class ObservationNormalizationConfig:
    """Normalization and termination parameters extracted from the legacy env."""

    radar_max_dist: float = 10.0
    u_max: float = 2.0
    v_max: float = 2.0
    w_max: float = 2.0
    p_max: float = 1.0
    q_max: float = 1.0
    r_max: float = 1.0
    max_attitude: float = np.pi / 2
    dist_goal_reached_tol: float = 1.3
    max_dist_from_goal: float = 120.0
    max_timesteps: int = 4000


@dataclass
class NavigationState:
    """Minimal navigation state extracted from the current environment."""

    position: np.ndarray
    attitude_rpy: np.ndarray
    relative_velocity: np.ndarray
    angular_velocity: np.ndarray
    goal_location: Optional[np.ndarray] = None
    goal_heading: float = 0.0
    last_attitude_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    last_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def update_navigation_errors(self) -> dict[str, float]:
        """Compute distance and heading errors for the current goal."""
        if self.goal_location is None:
            raise ValueError("goal_location must be set before computing errors")

        diff = self.goal_location - self.position
        delta_d = float(np.linalg.norm(diff))
        delta_theta = float(
            self.attitude_rpy[1] + wrap_to_pi(np.arctan2(diff[2], np.linalg.norm(diff[:2])))
        )
        delta_psi = float(wrap_to_pi(np.arctan2(diff[1], diff[0]) - self.attitude_rpy[2]))
        delta_heading_goal = float(wrap_to_pi(self.goal_heading - self.attitude_rpy[2]))
        stable_delta_theta = float(self.attitude_rpy[1] - self.last_attitude_rpy[1])
        distance_moved_per_step = float(np.linalg.norm(self.position - self.last_position))
        return {
            "delta_d": delta_d,
            "delta_theta": delta_theta,
            "delta_psi": delta_psi,
            "delta_heading_goal": delta_heading_goal,
            "stable_delta_theta": stable_delta_theta,
            "distance_moved_per_step": distance_moved_per_step,
        }


@dataclass
class RuntimeEpisodeState:
    """Mutable state needed to compute observations, rewards, and termination."""

    navigation: NavigationState
    observation_dim: int = 37
    reward_dim: int = 13
    continuous_reward_dim: int = 8
    t_steps: int = 0
    t_total_steps: int = 0
    episode: int = 0
    cumulative_reward: float = 0.0
    last_reward: float = 0.0
    done: bool = False
    goal_reached: bool = False
    collision: bool = False
    delta_d_history: list[float] = field(default_factory=list)
    last_reward_arr: np.ndarray = field(default_factory=lambda: np.zeros(13, dtype=np.float32))
    cumulative_reward_arr: np.ndarray = field(default_factory=lambda: np.zeros(13, dtype=np.float32))

    def sync_from(self, other: "RuntimeEpisodeState") -> None:
        """Mirror key transition state from another runtime state."""
        self.delta_d_history = other.delta_d_history.copy()
        self.last_reward_arr = other.last_reward_arr.copy()
        self.cumulative_reward_arr = other.cumulative_reward_arr.copy()
        self.navigation.last_attitude_rpy = other.navigation.last_attitude_rpy.copy()
        self.navigation.last_position = other.navigation.last_position.copy()
        if other.navigation.goal_location is not None:
            self.navigation.goal_location = other.navigation.goal_location.copy()
        self.navigation.goal_heading = other.navigation.goal_heading


class ObservationBuilder:
    """Build normalized observations from navigation and range data."""

    def __init__(self, config: ObservationNormalizationConfig) -> None:
        self.config = config

    def build(
        self,
        episode_state: RuntimeEpisodeState,
        navigation_errors: dict[str, float],
        reduced_radar_distances: np.ndarray,
    ) -> np.ndarray:
        """Build the normalized observation vector matching the legacy layout."""
        navigation = episode_state.navigation
        obs = np.zeros(episode_state.observation_dim, dtype=np.float32)

        delta_d = navigation_errors["delta_d"]
        obs[0] = np.clip(
            1 - (
                np.log(delta_d / self.config.max_dist_from_goal)
                / np.log(self.config.dist_goal_reached_tol / self.config.max_dist_from_goal)
            ),
            0,
            1,
        )
        obs[1] = np.clip(navigation_errors["delta_theta"] / (np.pi / 2), -1, 1)
        obs[2] = np.clip(navigation_errors["delta_psi"] / np.pi, -1, 1)

        obs[3] = np.clip(navigation.relative_velocity[0] / self.config.u_max, -1, 1)
        obs[4] = np.clip(navigation.relative_velocity[1] / self.config.v_max, -1, 1)
        obs[5] = np.clip(navigation.relative_velocity[2] / self.config.w_max, -1, 1)
        obs[6] = np.clip(navigation.attitude_rpy[0] / self.config.max_attitude, -1, 1)
        obs[7] = np.clip(navigation.attitude_rpy[1] / self.config.max_attitude, -1, 1)
        obs[8] = np.clip(np.sin(navigation.attitude_rpy[2]), -1, 1)
        obs[9] = np.clip(np.cos(navigation.attitude_rpy[2]), -1, 1)
        obs[10] = np.clip(navigation.angular_velocity[0] / self.config.p_max, -1, 1)
        obs[11] = np.clip(navigation.angular_velocity[1] / self.config.q_max, -1, 1)
        obs[12] = np.clip(navigation.angular_velocity[2] / self.config.r_max, -1, 1)
        obs[13:] = np.clip(reduced_radar_distances / self.config.radar_max_dist, 0, 1)
        return obs


class DoneEvaluator:
    """Evaluate legacy terminal conditions."""

    def __init__(self, config: ObservationNormalizationConfig) -> None:
        self.config = config

    def evaluate(
        self,
        episode_state: RuntimeEpisodeState,
        navigation_errors: dict[str, float],
    ) -> tuple[bool, list[int], list[bool]]:
        """Evaluate terminal conditions and return `(done, cond_idx, conditions)`."""
        conditions = [
            navigation_errors["delta_d"] < self.config.dist_goal_reached_tol,
            navigation_errors["delta_d"] > self.config.max_dist_from_goal,
            False,
            episode_state.t_steps >= self.config.max_timesteps,
            episode_state.collision,
        ]
        done = bool(np.any(conditions))
        if done and conditions[0]:
            episode_state.goal_reached = True
        cond_idx = [index for index, flag in enumerate(conditions) if flag]
        return done, cond_idx, conditions


class RewardMath:
    """Reward helpers migrated from the legacy environment."""

    @staticmethod
    def log_precision(x: float, x_goal: float, x_max: float) -> float:
        epsilon = 0.001
        return 1 - np.clip(
            np.log(max(x, epsilon) / x_max) / np.log(max(x_goal, epsilon) / x_max),
            0,
            1,
        )

    @staticmethod
    def cont_goal_constraints(
        x: float,
        delta_d: float,
        x_des: float,
        delta_d_des: float,
        x_max: float,
        delta_d_max: float,
        x_exp: float = 1.0,
        delta_d_exp: float = 1.0,
        x_rev: bool = False,
        delta_d_rev: bool = False,
    ) -> float:
        r_x = np.abs((float(x_rev) - RewardMath.log_precision(x, x_des, x_max))) ** x_exp
        r_delta_d = np.abs(
            (float(delta_d_rev) - RewardMath.log_precision(delta_d, delta_d_des, delta_d_max))
        ) ** delta_d_exp
        return r_x * r_delta_d

    @staticmethod
    def angle_control(r: float, r_goal: float, r_max: float, r_partition: float, r_exp: float) -> float:
        r_1 = RewardMath.log_precision(r, r_goal, r_partition) ** r_exp
        r_2 = RewardMath.log_precision(r, r_partition, r_max) ** r_exp
        return 1 - r_1 - r_2

    @staticmethod
    def beta_oa(theta_r, psi_r, theta_max, psi_max, epsilon_oa):
        return (1 - np.abs(theta_r) / theta_max) * (1 - np.abs(psi_r) / psi_max) + epsilon_oa

    @staticmethod
    def c_oa(d_r, d_max):
        return np.clip(1 - (d_r - 1) / d_max, 0, 1)

    @staticmethod
    def obstacle_avoidance_improve(
        theta_r: np.ndarray,
        psi_r: np.ndarray,
        d_r: np.ndarray,
        theta_max: float,
        psi_max: float,
        d_max: float,
        gamma_c: float,
        epsilon_c: float,
        epsilon_oa: float = 0.01,
        d_c: float = 3.0,
        tau_c: float = 1.0,
    ) -> float:
        beta = RewardMath.beta_oa(theta_r, psi_r, theta_max, psi_max, epsilon_oa)
        d_pen = np.exp(np.maximum(tau_c * (d_c - d_r), 0)) - 1
        beta += d_pen
        c = RewardMath.c_oa(d_r, d_max)
        return np.sum(beta) / (np.maximum((gamma_c * (1 - c)) ** 2, epsilon_c) @ beta) - 1


class BaseSimulationAdapter:
    """Shared simulation math and runtime primitives.

    This still does not open HoloOcean directly, but now contains enough of the
    old environment core that future runtime adapters can stop depending on the
    legacy `pierharbor_hovering.py` implementation.
    """

    def __init__(self, config: ObservationNormalizationConfig | None = None) -> None:
        self.config = config or ObservationNormalizationConfig()
        self.action_mapper = DEFAULT_ACTION_MAPPER
        self.rangefinder = RangeFinderGrid(max_dist=self.config.radar_max_dist)
        self.observation_builder = ObservationBuilder(self.config)
        self.done_evaluator = DoneEvaluator(self.config)

    def action_to_command(self, action: np.ndarray) -> np.ndarray:
        """Translate a low-level action vector to a thruster command."""
        return self.action_mapper.to_command(action)

    @staticmethod
    def angle_conversion(angle_deg: np.ndarray) -> np.ndarray:
        """Convert HoloOcean roll-pitch-yaw degrees to radians."""
        return np.asarray(angle_deg, dtype=np.float32) / 180.0 * np.pi

    @staticmethod
    def update_body_collision(radar_dist: np.ndarray) -> bool:
        """Return whether any reduced radar ray indicates direct collision."""
        return bool(np.any(radar_dist == 0))
