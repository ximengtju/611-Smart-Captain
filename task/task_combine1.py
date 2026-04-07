from env.pierharbor_hovering import BaseEnvironment
from env.pierharbor_hovering import BASE_CONFIG, Reward
from task.task1 import Navigation
from task.task2 import Obstacles_Avoidance
from typing import Tuple
from skimage.measure import block_reduce
import holoocean
import numpy as np
import gymnasium as gym

class MAInterface():
    def __init__(self, env_config: dict = BASE_CONFIG, mode: int = 0):
        self.config = env_config
        self.scenario_cfg = self.config["auv_config"]
        self.auv = holoocean.make(scenario_cfg=self.scenario_cfg, show_viewport=True)
        self.task_env = []

        task1_env = Navigation(env_config, self.auv, False)
        task2_env = Obstacles_Avoidance(env_config, self.auv, False)
        self.task_env.append(task1_env)
        self.task_env.append(task2_env)

        self.mode = 0


    def reset(self, seed = None, return_info = True, options = None):
        return self.task_env[self.mode].reset(seed, return_info, options)
        
    def step(self, action):
        return self.task_env[self.mode].step(action)
    
    def set_multi_mode_index(self, index: int):
        last_mode = self.mode
        self.mode = index
        self.task_env[self.mode].sync_state_from(self.task_env[last_mode])