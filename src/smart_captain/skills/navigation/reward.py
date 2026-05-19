"""Task1 navigation reward calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smart_captain.simulation.core.base_env import RewardMath


@dataclass(frozen=True)
class NavigationRewardBreakdown:
    """Named reward components used by the Task1 environment."""

    progress: float
    pitch_alignment: float
    yaw_alignment: float
    roll_stability: float
    pitch_rate_stability: float
    angular_rate_stability: float
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
        reward[5] = self.angular_rate_stability
        # Slot 6 is the legacy obstacle-avoidance reward. Navigation leaves it zero.
        reward[7] = self.action_penalty
        reward[8:] = self.terminal.astype(np.float32)
        return reward


def compute_task1_reward_breakdown(env, action: np.ndarray) -> NavigationRewardBreakdown:
    """Compute the Task1 reward terms from the current environment state."""
    del action

    if len(env.delta_d_list) == 0:
        env.delta_d_list.extend([env.delta_d, env.delta_d])
    elif len(env.delta_d_list) == 1:
        env.delta_d_list.append(env.delta_d)

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

    terminal = np.asarray(env.conditions, dtype=np.float32) * env.w_done
    if env.goal_reached:
        terminal[0] += 0.1 * (env.max_timesteps - env.t_steps)

    return NavigationRewardBreakdown(
        progress=float(progress),
        pitch_alignment=float(pitch_alignment),
        yaw_alignment=float(yaw_alignment),
        roll_stability=float(roll_stability),
        pitch_rate_stability=float(pitch_rate_stability),
        angular_rate_stability=0.0,
        action_penalty=0.0,
        terminal=terminal,
    )


def compute_task1_reward(env, action: np.ndarray) -> float:
    """Update env reward arrays and return the scalar Task1 reward."""
    breakdown = compute_task1_reward_breakdown(env, action)
    reward_arr = breakdown.as_array()
    env.last_reward_arr[:] = reward_arr
    env.cum_reward_arr[:] = env.cum_reward_arr + reward_arr
    reward = float(np.sum(reward_arr))
    env.last_reward = reward
    env.episode_state.last_reward = reward
    env.delta_d_xy_last = float(np.linalg.norm((env.goal_location - env.auv_position)[:2]))
    return reward


__all__ = [
    "NavigationRewardBreakdown",
    "compute_task1_reward",
    "compute_task1_reward_breakdown",
]
