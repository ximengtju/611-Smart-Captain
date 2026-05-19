"""Task4 obstacle-avoidance reward calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smart_captain.simulation.core.base_env import RewardMath


@dataclass(frozen=True)
class ObstacleRewardBreakdown:
    """Named reward components used by the Task4 environment."""

    progress: float
    pitch_alignment: float
    yaw_alignment: float
    roll_stability: float
    pitch_rate_stability: float
    depth_corridor: float
    obstacle_avoidance: float
    action_penalty: float
    terminal: np.ndarray

    def as_array(self) -> np.ndarray:
        """Return the 13-element reward vector used by legacy logs."""
        reward = np.zeros(13, dtype=np.float32)
        reward[0] = self.progress
        reward[1] = self.pitch_alignment
        reward[2] = self.yaw_alignment
        reward[3] = self.roll_stability
        reward[4] = self.pitch_rate_stability
        reward[5] = self.depth_corridor
        reward[6] = self.obstacle_avoidance
        reward[7] = self.action_penalty
        reward[8:] = self.terminal.astype(np.float32)
        return reward


def compute_goal_sector_obstacle_penalty(env) -> float:
    """Compute the Task4 obstacle penalty from reduced radar distances."""
    beta = env.radar.beta_reduced
    distances = env.radar_intersec_dist
    goal_mask = np.abs(env.ssa(beta - env.delta_psi)) < np.deg2rad(30)

    d_goal = float(np.min(distances[goal_mask])) if np.any(goal_mask) else float(env.radar_max_dist)
    d_min = float(np.min(distances))

    r_block_goal = -2.0 * max(0.0, (5.0 - d_goal) / 5.0) ** 2
    r_too_close = -3.0 * max(0.0, (3.0 - d_min) / 3.0) ** 2

    speed = float(np.linalg.norm(env.auv_relative_velocity))
    r_stuck = -1.0 if d_min < 2.0 and speed < 0.2 else 0.0

    return env.reward_factors["w_oa"] * (r_block_goal + r_too_close + r_stuck)


def compute_task4_reward_breakdown(env, action: np.ndarray) -> ObstacleRewardBreakdown:
    """Compute the Task4 reward terms from the current environment state."""
    del action

    if len(env.delta_d_list) < 2:
        env.delta_d_list.extend([env.delta_d, env.delta_d])

    delta_d_step = env.delta_d_list[-1] - env.delta_d_list[-2]
    progress = -delta_d_step / max(env.dt, 1e-6)

    pitch_alignment = -env.reward_factors["w_delta_theta"] * RewardMath.cont_goal_constraints(
        x=np.abs(env.delta_theta),
        delta_d=env.delta_d,
        x_des=0.0,
        delta_d_des=env.dist_goal_reached_tol,
        x_max=np.pi / 2,
        delta_d_max=env.max_dist_from_goal,
        x_exp=2.0,
        delta_d_exp=0.0,
        x_rev=False,
        delta_d_rev=False,
    )

    yaw_alignment = env.reward_factors["w_delta_psi"] * RewardMath.angle_control(
        r=np.abs(env.delta_psi),
        r_goal=0.0,
        r_max=np.pi,
        r_partition=np.pi / 6,
        r_exp=2,
    )

    roll_stability = -env.reward_factors["w_phi"] * (env.auv_attitude[0] / (np.pi / 2)) ** 2
    pitch_rate_stability = -env.reward_factors["w_theta"] * (
        (env.auv_attitude[1] - env.auv_last_attitude[1]) / (np.pi / 2)
    ) ** 2

    terminal = np.array(env.conditions, dtype=np.float32) * env.w_done
    if env.goal_reached:
        terminal[0] += 0.05 * (env.max_timesteps - env.t_steps)

    return ObstacleRewardBreakdown(
        progress=float(progress),
        pitch_alignment=float(pitch_alignment),
        yaw_alignment=float(yaw_alignment),
        roll_stability=float(roll_stability),
        pitch_rate_stability=float(pitch_rate_stability),
        depth_corridor=0.0,
        obstacle_avoidance=float(compute_goal_sector_obstacle_penalty(env)),
        action_penalty=0.0,
        terminal=terminal,
    )


def compute_task4_reward(env, action: np.ndarray) -> float:
    """Update env reward arrays and return the scalar Task4 reward."""
    breakdown = compute_task4_reward_breakdown(env, action)
    reward_arr = breakdown.as_array()
    env.last_reward_arr[:] = reward_arr
    env.cum_reward_arr = env.cum_reward_arr + reward_arr
    reward = float(np.sum(reward_arr))
    env.last_reward = reward
    env.episode_state.last_reward = reward
    env.delta_d_xy_last = float(np.linalg.norm((env.goal_location - env.auv_position)[:2]))
    return reward


__all__ = [
    "ObstacleRewardBreakdown",
    "compute_goal_sector_obstacle_penalty",
    "compute_task4_reward",
    "compute_task4_reward_breakdown",
]
