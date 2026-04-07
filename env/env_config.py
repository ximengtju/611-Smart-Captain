import numpy as np
import os
import copy

REGISTRATION_DICT = {
    "task1-v0": "task.task1:Navigation",
    "task2-v0": "task.task2:Obstacles_Avoidance"
}

BASE_CONFIG = {
    # === 仿真环境配置 ===
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
                            "UseRPY": True
                        }
                    },
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor0",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": 0,
                            "LaserDebug": False
                        }
                    },  
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor1",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": 15,
                            "LaserDebug": False
                        }
                    }, 
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor2",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": 30,
                            "LaserDebug": False
                        }
                    }, 
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor3",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": 45,
                            "LaserDebug": False
                        }
                    }, 
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor4",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": 60,
                            "LaserDebug": False
                        }
                    }, 
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor5",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": -15,
                            "LaserDebug": False
                        }
                    }, 
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor6",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": -30,
                            "LaserDebug": False
                        }
                    }, 
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor7",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": -45,
                            "LaserDebug": False
                        }
                    }, 
                    {
                        "sensor_type": "RangeFinderSensor",
                        "sensor_name": "RangeFinderSensor8",
                        "socket": "COM",
                        "configuration": {
                            "LaserMaxDistance": 10,
                            "LaserCount": 24,
                            "LaserAngle": -60,
                            "LaserDebug": False
                        }
                    }, 
                    {
                        "sensor_type": "VelocitySensor",
                        "socket": "IMUSocket"
                    },  
                ],
                "control_scheme": 0,
                "location": [0, 0, -10],
                "rotation": [0.0, 0.0, 0.0]
            }
        ]
    }, 

    # === 日志与存储 ===
    "save_path_folder": os.path.join(os.getcwd(), "logs"),
    "title": "DEFAULT",
    "verbose": 1,               # 是否输出日志到控制台 (0/1)
    "log_level": 30,             # 日志级别 (10=DEBUG,20=INFO,30=WARNING)
    "interval_episode_log": 10,  # 每多少回合打印一次日志

    # === 观测与动作空间维度 ===
    "n_observations": 37,        # 观测向量维度 (根据代码注释)
    "n_actions": 3,              # 动作向量维度 (根据实际AUV控制量设定)

    # === 传感器参数 ===
    "radar.max_dist": 10.0,       # 雷达最大探测距离 (应与LaserMaxDistance一致)

    # === AUV状态限幅 (归一化用) ===
    "u_max": 2.0,                 # 纵向速度最大值 (m/s)
    "v_max": 2.0,                 # 横向速度最大值 (m/s)
    "w_max": 2.0,                 # 垂向速度最大值 (m/s)
    "p_max": 1.0,                 # 横滚角速度最大值 (rad/s)
    "q_max": 1.0,                 # 俯仰角速度最大值 (rad/s)
    "r_max": 1.0,                 # 偏航角速度最大值 (rad/s)
    "max_attitude": np.pi/2,         # 最大允许姿态角 (rad)，90度

    # === 终止条件阈值 ===
    "max_timesteps": 4000,        # 每回合最大步数
    "interval_datastorage": 10,   # 数据存储间隔（步数）
    "dist_goal_reached_tol": 1.3, # 到达目标点的距离容差 (m)
    "velocity_goal_reached_tol": 0.1,   # 到达目标时的速度容差 (m/s)
    "ang_rate_goal_reached_tol": 0.1,   # 到达目标时的角速度容差 (rad/s)
    "attitude_goal_reached_tol": 0.1,   # 到达目标时的姿态容差 (rad)
    "max_dist_from_goal": 120.0,         # 最大允许距离误差 (超出即终止)

    # === 雷达相关（用于避障奖励） ===
    "radar.beta_max": 2 * np.pi,  # 雷达波束最大覆盖角度 (rad)，若为全向则设为2π

    # === 奖励权重因子 ===
    "reward_factors": {
        # 连续奖励分量权重
        "w_delta_theta": 1.0,      # 俯仰角误差惩罚因子
        "w_delta_psi": 1.0,        # 偏航角误差惩罚因子
        "w_phi": 1.0,              # 横滚角惩罚因子
        "w_theta": 0.0,            # 俯仰角变化率惩罚因子（代码中当前为0）
        "w_oa": 1.0,               # 避障奖励因子
        # 离散终止奖励权重（用于w_done）
        "w_goal": 100.0,           # 到达目标奖励
        "w_deltad_max": -10.0,      # 超出最大距离惩罚
        "w_Theta_max": -10.0,       # 超出最大角度惩罚（注意大小写）
        "w_t_max": -1.0,            # 超时惩罚
        "w_col": -100.0             # 碰撞惩罚
    },

    # === 动作奖励因子（当前未使用，保留占位） ===
    "action_reward_factors": 0.0,

    # === 导航任务参数 ===
    "r_min": 30.0,
    "r_max": 40.0
}