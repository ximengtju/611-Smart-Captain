from client.src.holoocean import holoocean
import numpy as np
import datetime
import logging
import time
import os
import pprint
import gymnasium as gym
from gymnasium.utils import seeding
from typing import Tuple, Optional, Union
from timeit import default_timer as timer
from env.env_config import BASE_CONFIG
from skimage.measure import block_reduce

logger = logging.getLogger(__name__)

class BaseEnvironment(gym.Env):
    def __init__(self, config: dict = BASE_CONFIG, auv = None, train_mode = True):
        self.config = config

        # ===初始化AUV===
        self.scenario_cfg = self.config["auv_config"]
        if train_mode == True:
            self.AUV = holoocean.make(scenario_cfg=self.scenario_cfg, show_viewport=True) # 加载智能体
        else:
            self.AUV = auv

        # === 初始化日志系统 ===
        self.log_level = self.config["log_level"]
        self.save_path_folder = self.config["save_path_folder"]
        self.title = self.config["title"]
        self.verbose = self.config["verbose"]
        self.interval_episode_log = self.config["interval_episode_log"]

        utc_str = datetime.datetime.utcnow().strftime('%Y_%m_%dT%H_%M_%S')
        os.makedirs(self.save_path_folder, exist_ok=True)
        logging.basicConfig(level=self.log_level,
                            filename=os.path.join(self.save_path_folder, f"{utc_str}__{self.title}.log"),
                            format='[%(asctime)s] [%(levelname)s] [%(module)s] - [%(funcName)s]: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S'
                            )
        # 检查是否需要输出到控制台
        if self.verbose:
            logging.getLogger().addHandler(logging.StreamHandler())
        logging.Formatter.converter = time.gmtime  # 确保使用UTC时间
        logger.info('---------- Docking3d Gym Logger ----------')
        logger.info('---------- ' + utc_str + ' ----------')
        logger.info('---------- Initialize environment ----------')
        logger.info('Gym environment settings: \n ' + pprint.pformat(config))


        # ====== 各项变量初始化 ======
        # === 初始化导航误差变量 ===
        self.delta_d = 0  # 距当前目标距离
        self.delta_psi = 0 
        self.delta_theta = 0 
        self.delta_heading_goal = 0  
        self.delta_d_list = [] 
        self.delta_d_final = 0  # 距最终目标距离

        # === 用在reset函数的初始化部分 === 
        self.n_observations = self.config["n_observations"]
        self.n_actions = self.config["n_actions"]
        self.radar_max_dist = self.config["radar.max_dist"]
        self.observation = np.zeros(self.n_observations)
        obs_low = -np.ones(self.n_observations)
        obs_low[0] = 0
        obs_low[13:] = 0 # 雷达射线观测最小值设为0
        self.action_space = gym.spaces.Box(low=np.array([-40,-20,-10,-20]),
                                           high=np.array([40,20,10,20]),
                                           dtype=np.float32) 
        self.observation_space = gym.spaces.Box(low=obs_low,
                                           high=np.ones(self.n_observations),
                                           dtype=np.float32) # observation的张量大小应该是37
        self.meta_data_observation = [
            ["delta_d", "delta_theta", "delta_psi"],               # 导航误差
            ["u", "v", "w"],                                       # 线速度
            ["phi", "theta", "psi_sin", "psi_cos"],                # 姿态角
            ["p", "q", "r"],                                       # 角速度
            [f"ray_{i}" for i in range(24)] # 雷达射线
        ]
        self.radar = Radar()

        # === 初始化仿真时间相关变量 ===
        self.t_total_steps = 0  # 环境中运行的总步数
        self.t_steps = 0  # 当前回合的步数
        self.episode = 0  # 当前回合数
        self.max_timesteps = self.config["max_timesteps"] # 最大步数限制
        self.interval_datastorage = self.config["interval_datastorage"] # 数据存储间隔
        self.info = {}  # 仿真信息字典

        # === 环境状态标志 ===
        self.goal_reached = False  # 是否到达目标点
        self.collision = False  # 是否发生碰撞
        self.conditions = None  # 终止条件判断数组

        # === 状态设置 ===
        self.u_max = config["u_max"]
        self.v_max = config["v_max"]
        self.w_max = config["w_max"]
        self.p_max = config["p_max"]
        self.q_max = config["q_max"]
        self.r_max = config["r_max"]

        # === 奖励初始化 ===
        self.n_rewards = 13         # 奖励分量总数
        self.n_cont_rewards = 8     # 连续奖励分量数
        self.last_reward = 0        # 上一步的奖励
        self.last_reward_arr = np.zeros(self.n_rewards)  # 连续奖励分量数组
        self.cumulative_reward = 0  # 累计奖励
        self.cum_reward_arr = np.zeros(self.n_rewards)   # 离散奖励分量数组
        self.reward_factors = self.config["reward_factors"]  # 奖励权重字典
        self.w_done = np.array([
            self.reward_factors["w_goal"],        # 到达目标权重
            self.reward_factors["w_deltad_max"],  # 最大距离误差权重
            self.reward_factors["w_Theta_max"],   # 最大角度误差权重
            self.reward_factors["w_t_max"],       # 超时权重
            self.reward_factors["w_col"]          # 碰撞权重
        ])

        self.meta_data_reward = [
            "Nav_delta_d",          # 导航距离误差奖励
            "Nav_delta_theta",      # 俯仰角误差奖励  
            "Nav_delta_psi",        # 航向角误差奖励
            "Att_phi",              # 横滚角奖励
            "Att_theta",            # 俯仰角奖励
            "Thetadot",             # 角速度奖励
            "obstacle_avoid",       # 避障奖励
            "action",               # 动作惩罚
            "Done-Goal_reached",    # 到达目标奖励
            "Done-out_pos",         # 位置超出惩罚
            "Done-out_att",         # 姿态超出惩罚
            "Done-max_t",           # 超时惩罚
            "Done-collision"        # 碰撞惩罚
        ]
        
        self.action_reward_factors = self.config["action_reward_factors"]  # 动作奖励因子

        self.radar_angles_deg = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 
                         180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
        self.radar_beta = np.deg2rad(self.radar_angles_deg)
        self.radar_beta = (self.radar_beta + np.pi) % (2 * np.pi) - np.pi # 水平传感器的角度情况
        self.radar_beta_max = np.pi # 水平传感器最大角度

        # === 终止条件设置 ===
        self.done = False
        self.meta_data_done = self.meta_data_reward[self.n_cont_rewards:]  # 终止条件元数据
        self.goal_constraints = []  # 目标约束条件列表
        self.goal_location = None   # 目标位置（在generate_environment中定义）
        self.dist_goal_reached_tol = self.config["dist_goal_reached_tol"]  # 到达目标的距离容差
        self.velocity_goal_reached_tol = self.config["velocity_goal_reached_tol"]  # 目标点速度限制
        self.ang_rate_goal_reached_tol = self.config["ang_rate_goal_reached_tol"]  # 目标点加速度限制
        self.attitude_goal_reached_tol = self.config["attitude_goal_reached_tol"]  # 目标点姿态容差
        self.max_dist_from_goal = self.config["max_dist_from_goal"]  # 最大允许距离
        self.max_attitude = self.config["max_attitude"]              # 最大允许姿态角
        self.heading_goal_reached = 0  # 目标点期望航向（俯仰角应为0）

        # === 仿真时间记录 ===
        self.start_time_sim = timer()

        # === 具身部分 ===



    def reset(self, seed: Optional[int] = None,
                return_info: bool = True,
                options: Optional[dict] = None,
                ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:

        return_info_dict = self.info.copy() # 存上个epoch数据

        # === auv更新 ===
        sensor_return = self.AUV.reset()
        dynamics_obs = sensor_return['DynamicsSensor']
        velocity_obs = sensor_return['VelocitySensor']

        # ===各项数据指标更新
        self.auv_last_attitude = np.zeros(3) 
        self.auv_last_position = np.zeros(3) 
        self.auv_attitude = dynamics_obs[15:18] 
        self.auv_attitude = self.angle_conversion(self.auv_attitude)
        self.auv_position = dynamics_obs[6:9]
        self.radar.update_from_sensor(sensor_return)

        self.t_steps = 0           
        self.goal_reached = False  
        self.collision = False     
        self.info = {}             

        self.observation = np.zeros(self.n_observations, dtype=np.float32)
        self.last_reward = 0                    # 重置上一步奖励
        self.cumulative_reward = 0              # 重置累计奖励
        self.last_reward_arr = np.zeros(self.n_rewards)  # 重置奖励分量数组
        self.cum_reward_arr = np.zeros(self.n_rewards)   # 重置累计奖励分量数组
        self.done = False                       # 重置终止标志
        self.conditions = None                  # 重置条件判断数组
        self.goal_constraints = []              # 清空目标约束条件列表

        self.delta_d = 0      # 重置距离误差
        self.delta_psi = 0    # 重置航向角误差  
        self.delta_theta = 0  # 重置俯仰角误差
        self.delta_d_list = [] # 重置距离误差储存

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed) # 创建新的随机数生成器
            np.random.seed(seed) # 设置numpy的随机种子
        
        if self.episode == 1 or self.episode % self.interval_episode_log == 0:
            logger.info("Environment reset call: \n" + pprint.pformat(return_info_dict))
        else:
            logger.debug("Environment reset call: \n" + pprint.pformat(return_info_dict))

        self.episode += 1 # 记录epoch

        self.generate_environment() # 目标点更新
        self.update_navigation_errors() # 差值更新
        self.delta_d_list.append(self.delta_d) # 距离差列表更新

        if return_info:
            return self.observation, return_info_dict
        return self.observation

    def generate_environment(self):
        self.goal_location = np.array([30,0,0])
        self.heading_goal_reached = 0


    def update_navigation_errors(self, location=None):
        if location is None:
            target_location = self.goal_location
        else:
            target_location = location
        
        diff = target_location - self.auv_position

        self.delta_d = np.linalg.norm(diff)
        self.delta_theta = self.auv_attitude[1] + (self.ssa(np.arctan2(diff[2], np.linalg.norm(diff[:2]))))
        self.delta_psi = self.ssa(np.arctan2(diff[1], diff[0]) - self.auv_attitude[2])

        self.delta_heading_goal = self.ssa(self.heading_goal_reached - self.auv_attitude[2])
        self.stable_delta_theta = self.auv_attitude[1] - self.auv_last_attitude[1]
        self.distance_moved_per_step = np.linalg.norm(self.auv_position - self.auv_last_position)
    

    def ssa(self, angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def angle_conversion(self, angle: np.ndarray) -> np.ndarray:
        return angle / 180.0 * np.pi

    def update_body_collision(self, radar_dist: np.ndarray) -> bool:
        return np.any(radar_dist == 0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:

        truncated = False
        command = self.action_to_command(action)
        sensor_return = self.AUV.step(command)

        # === 状态获取 ===
        dynamics_obs = sensor_return['DynamicsSensor']  # 基础数据
        velocity_obs = sensor_return['VelocitySensor'] # 速度数据
        self.auv_attitude = dynamics_obs[15:18] 
        self.auv_attitude = self.angle_conversion(self.auv_attitude)
        self.auv_position = dynamics_obs[6:9]
        self.auv_relative_velocity = velocity_obs
        self.auv_angular_velocity = dynamics_obs[12:15] 
        self.radar.update_from_sensor(sensor_return)
        self.radar_intersec_dist = self.radar.intersec_dist_reduced
        
        # === 数据处理 ===
        self.collision = self.update_body_collision(self.radar_intersec_dist) # 碰撞检测
        self.update_navigation_errors() # 更新导航误差
        self.delta_d_list.append(self.delta_d)
        self.observation = self.observe() # 处理输出数据
        self.done, cond_idx = self.is_done() # 结束判断
        self.last_reward = self.reward_step_impro(action)

        # === 状态数据的更新与储存 ===
        self.cumulative_reward += self.last_reward
        self.t_total_steps += 1
        self.t_steps += 1
        self.auv_last_attitude = self.auv_attitude
        self.auv_last_position = self.auv_position
        self.last_radar_r = self.radar_intersec_dist
        self.info = {
            "episode_number": self.episode,        # 当前回合编号（注意：sb3库使用episode作为关键字）
            "t_step": self.t_steps,                # 当前回合的步数
            "t_total_steps": self.t_total_steps,   # 总步数（所有回合累计）
            "cumulative_reward": self.cumulative_reward,  # 当前回合累计奖励
            "last_reward": self.last_reward,       # 这一步获得的奖励
            "done": self.done,                     # 回合是否结束
            "conditions_true": cond_idx,           # 满足的终止条件索引数组
            "conditions_true_info": [self.meta_data_done[i] for i in cond_idx],  # 终止条件描述
            "collision": self.collision,           # 是否发生碰撞
            "goal_reached": self.goal_reached,     # 是否到达目标
            # "goal_constraints": self.goal_constraints,  # 注释掉的目标约束条件
            "delta_d": self.delta_d,               # 当前距离误差
            "position": self.auv_position          # 机器人当前位置
        }

        if self.t_total_steps==1999:
            print(f"auv_position:({self.auv_position}")

        return self.observation, self.last_reward, self.done, truncated, self.info
    
    def action_to_command(self, action: np.ndarray):
        command = np.zeros(8)
        command[4:8] += action[0] # 前后

        command[4] += action[1] # 左右平移
        command[5] -= action[1]
        command[6] += action[1] 
        command[7] -= action[1] 

        command[4] += action[2] # 左右       
        command[5] -= action[2]           
        command[6] -= action[2]           
        command[7] += action[2]          

        command[0:4] = action[3] # 上下

        return command

    def observe(self):
        obs = np.zeros(self.n_observations, dtype=np.float32)

        # === 目标状态归一化 ===
        obs[0] = np.clip(1 - (np.log(self.delta_d / self.max_dist_from_goal) / np.log(
            self.dist_goal_reached_tol / self.max_dist_from_goal)), 0, 1)
        obs[1] = np.clip(self.delta_theta / (np.pi / 2), -1, 1)
        obs[2] = np.clip(self.delta_psi / np.pi, -1, 1)

        # === AUV状态归一化 ===
        obs[3] = np.clip(self.auv_relative_velocity[0] / self.u_max, -1, 1)  
        obs[4] = np.clip(self.auv_relative_velocity[1] / self.v_max, -1, 1)  
        obs[5] = np.clip(self.auv_relative_velocity[2] / self.w_max, -1, 1)  
        obs[6] = np.clip(self.auv_attitude[0] / self.max_attitude, -1, 1)  
        obs[7] = np.clip(self.auv_attitude[1] / self.max_attitude, -1, 1)  
        obs[8] = np.clip(np.sin(self.auv_attitude[2]), -1, 1)  
        obs[9] = np.clip(np.cos(self.auv_attitude[2]), -1, 1)  
        obs[10] = np.clip(self.auv_angular_velocity[0] / self.p_max, -1, 1)  
        obs[11] = np.clip(self.auv_angular_velocity[1] / self.q_max, -1, 1)  
        obs[12] = np.clip(self.auv_angular_velocity[2] / self.r_max, -1, 1)  

        obs[13:] = np.clip(self.radar_intersec_dist / self.radar_max_dist, 0, 1) # 雷达归一化

        return obs

    def is_done(self):
        self.conditions = [
            self.delta_d < self.dist_goal_reached_tol,
            self.delta_d > self.max_dist_from_goal,
            False,
            self.t_steps >= self.max_timesteps,
            self.collision
        ]
    
        done = bool(np.any(self.conditions))
        if done:
            if self.conditions[0]:
                self.goal_reached = True
            if self.conditions[2]:
                print("Attitude too high, steps: ", self.t_steps)
        cond_idx = [i for i, x in enumerate(self.conditions) if x]
        return done, cond_idx
    
    def reward_step_impro(self, action: np.ndarray, dt=0.005):

        delta_d = self.delta_d_list[-1] - self.delta_d_list[-2]
        self.last_reward_arr[0] = - delta_d/dt

        self.last_reward_arr[1] = - self.reward_factors["w_delta_theta"] * Reward.cont_goal_constraints(
            x=np.abs(self.delta_theta),
            delta_d=self.delta_d,
            x_des=0.0,
            delta_d_des=self.dist_goal_reached_tol, # 当前为1.3，实际可调
            x_max=np.pi / 2,
            delta_d_max=self.max_dist_from_goal, # 当前为40，实际可调
            x_exp=4.0,
            delta_d_exp=0.0,
            x_rev=False,
            delta_d_rev=False
        )

        self.last_reward_arr[2] = self.reward_factors["w_delta_psi"] * Reward.angle_control(
            r=np.abs(self.delta_psi),
            r_goal=0.0,
            r_max=np.pi,
            r_patition=np.pi / 6,
            r_exp=4
        )

        self.last_reward_arr[3] = -self.reward_factors["w_phi"] * (self.auv_attitude[0]/ (np.pi / 2)) ** 2

        self.last_reward_arr[4] = -self.reward_factors["w_theta"]* 0 * (
            (self.auv_attitude[1]-self.auv_last_attitude[1])  / (np.pi / 2)) ** 2
        
        self.last_reward_arr[5] = 0 # 这里是横滚角的角速度惩罚，暂时不作定义

        self.last_reward_arr[6] = - self.reward_factors["w_oa"] * Reward.obstacle_avoidance_improve(
            theta_r=self.radar.alpha, psi_r=self.radar.beta, d_r=self.radar.intersec_dist,
            theta_max=self.radar.alpha_max, psi_max=self.radar.beta_max, d_max=self.radar.max_dist,
            gamma_c=1, epsilon_c=0.001, epsilon_oa=0.01)
        
        self.last_reward_arr[7] = 0 # 这里是动作幅度惩罚，但暂时置0
        
        self.last_reward_arr[self.n_cont_rewards:] = np.array(self.conditions) * self.w_done

        self.cum_reward_arr = self.cum_reward_arr + self.last_reward_arr

        reward = float(np.sum(self.last_reward_arr))

        return reward
    
    def sync_state_from(self, other):
        """从另一个同类型环境实例复制所有关键状态变量"""
        self.delta_d_list = other.delta_d_list.copy()
        self.last_reward_arr = other.last_reward_arr.copy()
        self.cum_reward_arr = other.cum_reward_arr.copy()
        self.auv_last_attitude = other.auv_last_attitude.copy()
        self.auv_last_position = other.auv_last_position.copy()
        self.last_radar_r = other.last_radar_r.copy() if hasattr(other, 'last_radar_r') else None
        # 还可复制目标位置等
        self.goal_location = other.goal_location.copy()
        self.heading_goal_reached = other.heading_goal_reached
    


class Reward:
    @staticmethod
    def log_precision(x: float, x_goal: float, x_max: float) -> float:
        epsilon = 0.001  # Protection against log(0.0)
        return 1 - np.clip((np.log(max(x, epsilon) / x_max) / np.log(max(x_goal, epsilon) / x_max)), 0, 1)
    
    @staticmethod
    def cont_goal_constraints(x: float, delta_d: float, x_des: float, delta_d_des: float, x_max: float,
                              delta_d_max: float, x_exp: float = 1.0, delta_d_exp: float = 1.0, x_rev: bool = False,
                              delta_d_rev: bool = False) -> float:
        r_x = np.abs((float(x_rev) - Reward.log_precision(x, x_des, x_max))) ** x_exp
        r_delta_d = np.abs(
            (float(delta_d_rev) - Reward.log_precision(delta_d, delta_d_des, delta_d_max))) ** delta_d_exp
        return r_x * r_delta_d
    
    @staticmethod
    def angle_control(r, r_goal, r_max, r_patition, r_exp):
        r_1 = Reward.log_precision(r, r_goal, r_patition) ** r_exp
        r_2 = Reward.log_precision(r, r_patition, r_max) ** r_exp
        return 1 - r_1 - r_2
    
    @staticmethod
    def obstacle_avoidance_improve(theta_r: np.ndarray, psi_r: np.ndarray, d_r: np.ndarray, theta_max: float, psi_max: float,
                           d_max: float, gamma_c: float, epsilon_c: float, epsilon_oa: float = 0.01, 
                           d_c: float = 3.0, tau_c: float = 1.0):
        """
        修改版避障奖励，核心在于，任意射线与障碍物的距离实质上小于1时，都应当远离，不管是哪个方向的射线
        """
        beta = Reward.beta_oa(theta_r, psi_r, theta_max, psi_max, epsilon_oa)
        d_pen = np.exp(np.maximum(tau_c *(d_c - d_r), 0)) - 1
        beta += d_pen
        c = Reward.c_oa(d_r, d_max)
        return np.sum(beta) / (np.maximum((gamma_c * (1 - c)) ** 2, epsilon_c) @ beta) - 1
    
    @staticmethod
    def beta_oa(theta_r, psi_r, theta_max, psi_max, epsilon_oa):
        a = (1 - np.abs(theta_r) / theta_max) * (1 - np.abs(psi_r) / psi_max) + epsilon_oa
        return a

    @staticmethod
    def c_oa(d_r, d_max):
        return np.clip(1 - (d_r - 1) / d_max, 0, 1)
    


class Radar:
    """
    雷达类，根据9个垂直方向、24个水平方向的测距传感器构建。
    """
    def __init__(self, blocksize_reduce=3):
        deg2rad = np.pi / 180.0
        
        self.vertical_angles = np.array([-60, -45, -30, -15, 0, 15, 30, 45, 60]) * deg2rad
        
        self.radar_angles_deg = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 
                         180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
        self.horizontal_angles = np.deg2rad(self.radar_angles_deg)
        self.horizontal_angles = (self.horizontal_angles + np.pi) % (2 * np.pi) - np.pi

        self.n_vertical = len(self.vertical_angles)        # 9
        self.n_horizontal = len(self.horizontal_angles)    # 24
        self.n_rays = self.n_vertical * self.n_horizontal   # 216

        # 构建每条射线的垂直角度数组：每个垂直角度重复 n_horizontal 次
        self.alpha = np.repeat(self.vertical_angles, self.n_horizontal)
        self.alpha_max = np.pi / 3
        # 构建每条射线的水平角度数组：水平角度整体平铺 n_vertical 次
        self.beta = np.tile(self.horizontal_angles, self.n_vertical)
        self.beta_max = np.pi

        # 初始化探测距离（先填充0，后续通过 update 设置）
        self.intersec_dist = np.zeros(self.n_rays)
        self.max_dist = 10.0  # 假设最大探测距离，可根据实际情况设定

        # 降采样参数
        self.blocksize_reduce = blocksize_reduce

    def update_from_sensor(self, sensor_return):
        """
        从传感器返回数据中提取各射线距离，更新 intersec_dist。
        sensor_return 是一个字典，键为传感器名称，值为24维数组。
        传感器名称与垂直角度的对应关系：
        'RangeFinderSensor0' -> 0°
        'RangeFinderSensor1' -> 15°
        'RangeFinderSensor2' -> 30°
        'RangeFinderSensor3' -> 45°
        'RangeFinderSensor4' -> 60°
        'RangeFinderSensor5' -> -15°
        'RangeFinderSensor6' -> -30°
        'RangeFinderSensor7' -> -45°
        'RangeFinderSensor8' -> -60°
        """
        # 建立垂直角度到传感器键的映射（按角度排序后对应的键）
        angle_to_key = {
            -np.pi/3: 'RangeFinderSensor8',
            -np.pi/4: 'RangeFinderSensor7',
            -np.pi/6: 'RangeFinderSensor6',
            -np.pi/12: 'RangeFinderSensor5',
            0:   'RangeFinderSensor0',
            np.pi/12:  'RangeFinderSensor1',
            np.pi/6:  'RangeFinderSensor2',
            np.pi/4:  'RangeFinderSensor3',
            np.pi/3:  'RangeFinderSensor4'
        }

        # 按垂直角度顺序，将每个传感器的24维数组垂直堆叠成二维矩阵 (9, 24)
        dist_matrix = np.vstack([sensor_return[angle_to_key[angle]] for angle in self.vertical_angles])
        # 展平为一维数组，顺序与 self.alpha, self.beta 一致（先垂直后水平）
        self.intersec_dist = dist_matrix.flatten()

    @property
    def intersec_dist2d(self):
        """将一维距离重塑为二维矩阵 (垂直, 水平)"""
        return self.intersec_dist.reshape((self.n_vertical, self.n_horizontal))

    @property
    def intersec_dist_reduced(self):
        """
        对二维距离矩阵进行块降采样：块大小为 blocksize_reduce×blocksize_reduce，
        每个块内取中位数，然后将结果展平为一维数组。
        """
        # 使用 skimage.measure.block_reduce，边界不足的块自动保留
        reduced = block_reduce(self.intersec_dist2d,
                               block_size=(self.blocksize_reduce, self.blocksize_reduce),
                               func=np.median)
        return reduced.flatten()
