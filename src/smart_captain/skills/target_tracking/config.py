from __future__ import annotations

import numpy as np
import os
import copy

from smart_captain.skills.base import SkillSpec


MPC_CONFIG = {
    # === 基础配置 ===
    "m": 31.02 ,
    "g": 9.81,
    "rho": 1000.0,
    "V": 0.03554577,
    "dot_t": 0.005,

    "r_G": np.array([-0.059, 0.0046, -0.0282]),
    "r_B": np.array([-0.0596, 0.0029, -0.0185]),
    "d_lin": 1.0,
    "d_ang": 0.75,

    "I": np.diag([0.53, 0.67, 1.15]),
    "thruster_pos": np.array([
            [ 0.1818,  0.2214, -0.0400],
            [ 0.1818, -0.2214, -0.0400],
            [-0.3143, -0.2214, -0.0400],
            [-0.3143,  0.2214, -0.0400],
            [ 0.0739,  0.1823, -0.0021],
            [ 0.0739, -0.1823, -0.0021],
            [-0.2064, -0.1823, -0.0021],
            [-0.2064,  0.1823, -0.0021]
        ]),
    "horizon": 50,
    "v_cruise": 0.5,
    "Q": np.diag([50.0, 50.0, 50.0,   # x,y,z
                    1.0, 1.0, 1.0,     # phi,theta,psi
                    5.0, 5.0, 5.0,     # u,v,w
                    0.0, 0.0, 0.0]),
    "R": np.diag([0.001] * 4),
    "Rd": np.diag([0.001] * 4),
    "u_min": np.array([-40, -5, -20, -5]),
    "u_max": np.array([40, 5, 20, 5]),
    "x_min": np.array([-1000.0, -1000.0, -1000.0, 0, 0, 0, -2.0, -2.0, -2.0, -3.14, -3.14, -3.14]),
    "x_max": np.array([1000.0, 1000.0, 1000.0, np.pi, np.pi, np.pi, 2.0, 2.0, 2.0, 3.14, 3.14, 3.14]),

    "obs_space": 12,
    "command_space": 4,
}

TARGET_TRACKING_SPEC = SkillSpec(
    name="target_tracking",
    env_cls="smart_captain.skills.target_tracking.env:TargetTrackingEnv",
    policy_cls="smart_captain.skills.target_tracking.policy:TargetTrackingPolicy",
    default_scenario="pier_harbor",
    observation_dim=12 + (MPC_CONFIG["horizon"] + 1) * 12,
    action_dim=4,
    description="MPC-based target tracking skill.",
    default_sensors=("dynamics", "velocity", "rangefinder"),
    tags=("target_tracking", "mpc", "trajectory"),
    config_entrypoint="smart_captain.skills.target_tracking.config:TARGET_TRACKING_SPEC",
    metadata={
        "controller": "mpc",
        "action_order": ("surge", "sway", "heave", "yaw"),
    },
)