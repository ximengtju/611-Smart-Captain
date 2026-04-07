from env.pierharbor_hovering import BaseEnvironment
from env.pierharbor_hovering import BASE_CONFIG, Reward
from typing import Tuple
from skimage.measure import block_reduce
import numpy as np
import gymnasium as gym

class MAInterface(BaseEnvironment):
    def __init__(self, env_config: dict = BASE_CONFIG, mode: int = 0):
        super().__init__(env_config)
        self.r_min = env_config["r_min"]
        self.r_max = env_config["r_max"]
        self.mode = mode

        # === mode2的初始化部分 ===
        obs_low = -np.ones(40)
        obs_low[0] = 0
        obs_low[13:] = 0 # 雷达射线观测最小值设为0
        if self.mode == 1:
            self.observation_space = gym.spaces.Box(low=obs_low,
                                            high=np.ones(40),
                                            dtype=np.float32)
        
        self.vertical_angles = np.array([-60, -45, -30, -15, 0, 15, 30, 45, 60]) * np.pi / 180.0
        
        self.radar_angles_deg = [0, 15, 30, 45, 60, 300, 315, 330, 345]
        self.horizontal_angles = np.deg2rad(self.radar_angles_deg)
        self.horizontal_angles = (self.horizontal_angles + np.pi) % (2 * np.pi) - np.pi

        self.n_vertical = len(self.vertical_angles)        # 9
        self.n_horizontal = len(self.horizontal_angles)    # 9
        self.n_rays = self.n_vertical * self.n_horizontal   # 216

        self.alpha = np.repeat(self.vertical_angles, self.n_horizontal)
        self.alpha_max = np.pi / 3
        self.beta = np.tile(self.horizontal_angles, self.n_vertical)
        self.beta_max = np.pi / 3


    def generate_environment(self, reset_only_position: bool = False):
        super().generate_environment()

        psi_auv = self.auv_attitude[2]
        r_target = np.random.uniform(self.r_min, self.r_max)
        psi_target_rel = np.random.uniform(-np.pi, np.pi) 
        psi_target_abs = self.ssa(psi_auv + psi_target_rel) 

        x_target = r_target * np.cos(psi_target_abs)
        y_target = r_target * np.sin(psi_target_abs)
        z_target = np.random.uniform(-15.0, -5.0)
        self.goal_location = np.array([x_target, y_target, z_target])

    def reset(self, seed = None, return_info = True, options = None):
        if self.mode == 0:
            return super().reset(seed, return_info, options)
        
        elif self.mode == 1:
            _, return_info_dict = super().reset()
            observation = np.zeros(40, dtype=np.float32)
            return observation, return_info_dict
    
    def step(self, action):
        obs, reward, done, _, info = super().step(action)
        if self.mode == 0:
            return obs, reward, done, False, info
        elif self.mode == 1:
            self.intersec_dist = self.intersec_dist_reduced_mode1()
            self.obser = self.task_observe_mode1(obs)
            self.reward = self.reward_step_mode1(reward)
            return self.obser, self.reward, done, False, info
    
    def intersec_dist_reduced_mode1(self):
        raw_dist = self.radar.intersec_dist
        n_vert = self.radar.n_vertical    # 9
        n_horiz = self.radar.n_horizontal # 24

        dist_mat = raw_dist.reshape((n_vert, n_horiz))

        horiz_indices = [20, 21, 22, 23, 0, 1, 2, 3, 4]
        sub_mat = dist_mat[:, horiz_indices]
        self.dist = sub_mat.flatten()
        reduced = block_reduce(sub_mat, block_size=(3, 1), func=np.min)

        return reduced.flatten()

    def task_observe_mode1(self, obs):
        obser = np.zeros(40, dtype=np.float32)
        obser[0:13] = obs[0:13]
        obser[13:] = np.clip(self.intersec_dist / self.radar_max_dist, 0, 1) # 雷达归一化

        return obser
    
    def reward_step_mode1(self, reward):
        # 重新计算一遍奖励，之前的奖励无法在非BaseEnv中使用
        reward = reward - self.last_reward_arr[6]
        
        self.last_reward_arr[6] = - self.reward_factors["w_oa"] * Reward.obstacle_avoidance_improve(
            theta_r=self.alpha, psi_r=self.beta, d_r=self.dist,
            theta_max=self.alpha_max, psi_max=self.beta_max, d_max=self.radar.max_dist,
            gamma_c=1, epsilon_c=0.001, epsilon_oa=0.01)
        reward = reward + self.last_reward_arr[6]

        return reward
    
    def set_multi_mode_index(self, index: int):
        self.mode = index