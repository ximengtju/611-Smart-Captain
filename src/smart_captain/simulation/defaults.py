"""Default simulation configuration migrated from the legacy environment."""

from __future__ import annotations

import os

import numpy as np


DEFAULT_ENV_CONFIG = {
    "auv_config": {
        "name": "Hovering",
        "world": "PierHarbor",
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
                "location": [0, 0, -10],
                "rotation": [0.0, 0.0, 0.0],
            }
        ],
    },
    "save_path_folder": os.path.join(os.getcwd(), "logs"),
    "title": "DEFAULT",
    "verbose": 1,
    "log_level": 30,
    "interval_episode_log": 10,
    "n_observations": 37,
    "n_actions": 3,
    "radar.max_dist": 10.0,
    "u_max": 2.0,
    "v_max": 2.0,
    "w_max": 2.0,
    "p_max": 1.0,
    "q_max": 1.0,
    "r_max": 1.0,
    "max_attitude": np.pi / 2,
    "max_timesteps": 4000,
    "interval_datastorage": 10,
    "dist_goal_reached_tol": 1.3,
    "velocity_goal_reached_tol": 0.1,
    "ang_rate_goal_reached_tol": 0.1,
    "attitude_goal_reached_tol": 0.1,
    "max_dist_from_goal": 120.0,
    "radar.beta_max": 2 * np.pi,
    "reward_factors": {
        "w_delta_theta": 1.0,
        "w_delta_psi": 1.0,
        "w_phi": 1.0,
        "w_theta": 0.0,
        "w_oa": 1.0,
        "w_goal": 100.0,
        "w_deltad_max": -10.0,
        "w_Theta_max": -10.0,
        "w_t_max": -1.0,
        "w_col": -100.0,
    },
    "action_reward_factors": 0.0,
    "r_min": 30.0,
    "r_max": 40.0,
}
