"""Default simulation configuration migrated from the legacy environment."""

from __future__ import annotations

import os

import numpy as np


DEFAULT_ENV_CONFIG = {
    "auv_config": {
        "name": "Hovering",
        "world": "OpenWater",
        "package_name": "Ocean",
        "main_agent": "auv0",
        "ticks_per_sec": 200,
        "agents": [
            {
                "agent_name": "auv0",
                "agent_type": "HoveringAUV",
                "sensors": [
                    {
                        "sensor_type": "DynamicsSensor",
                        "socket": "COM",
                        "configuration": {
                            "UseCOM": True,
                            "UseRPY": True,
                        },
                    },
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor0",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": 0,
                            "LaserDebug": False,
                        },
                    },
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor1",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": 15,
                            "LaserDebug": False,
                        },
                    },
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor2",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": 30,
                            "LaserDebug": False,
                        },
                    },
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor3",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": 45,
                            "LaserDebug": False,
                        },
                    },
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor4",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": 60,
                            "LaserDebug": False,
                        },
                    },
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor5",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": -15,
                            "LaserDebug": False,
                        },
                    },
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor6",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": -30,
                            "LaserDebug": False,
                        },
                    },
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor7",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": -45,
                            "LaserDebug": False,
                        },
                    },
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor8",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": -60,
                            "LaserDebug": False,
                        },
                    },
                    {
                        "sensor_type": "VelocitySensor",
                        "socket": "IMUSocket",
                    },
                ],
                "control_scheme": 0,
                "location": [0,0,-50],
                "rotation": [0.0, 0.0, 0.0],
            }
        ],
    },
    "debug_trace.enabled": False,
    "debug_trace.episode_interval": 5,
    "debug_trace.step_interval": 50,
    "show_viewport": False,
    "save_path_folder": os.path.join(os.getcwd(), "logs"),
    "title": "DEFAULT",
    "verbose": 1,
    "log_level": 30,
    "interval_episode_log": 10,
    "n_observations": 37,
    "n_actions": 4,
    "radar.max_dist": 10.0,
    # 碰撞处理
    "collision_distance": 0.7,
    "collision_recovery_safe_distance": 8.0,
    # 零动作的控制时间
    "collision_recovery_stop_steps": 400,
    "collision_recovery_clear_steps": 100,
    "collision_recovery_max_steps": 3000,
    # 上浮强度
    "collision_recovery_heave": 16.0,

    "u_max": 4.0,
    "v_max": 4.0,
    "w_max": 3.0,
    "p_max": 1.0,
    "q_max": 1.0,
    "r_max": 1.0,
    "max_attitude": np.pi / 2,
    "max_timesteps": 20000,
    "interval_datastorage": 10,
    "dist_goal_reached_tol": 2.0,
    "velocity_goal_reached_tol": 0.2,
    "ang_rate_goal_reached_tol": 0.2,
    "attitude_goal_reached_tol": 0.2,
    "max_dist_from_goal": 150.0,
    "radar.beta_max": 2 * np.pi,
    "z_goal_reached_tol": 2.5,
    "depth_soft_tol": 6.0,
    "depth_gate_dist": 40.0,
    "depth_band_margin": 8.0,
    "reward_factors": {
        "w_delta_theta": 1.0,
        "w_delta_psi": 1.0,
        "w_phi": 0.2,
        "w_theta": 1.0,
        "w_oa": 1.0,
        "w_depth": 0.8,
        "w_w": 0.08,
        "w_heave": 0.02,
        "w_goal": 40.0,
        "w_deltad_max": -30.0,
        "w_Theta_max": -30.0,
        "w_t_max": -10.0,
        "w_col": -50.0,
    },
    "action_reward_factors": 0.0,
    "r_min":120,
    "r_max":130,
}

# open water location默认 111, -127, -281.0
# 150, -73, -285