import os
import numpy as np
import inspect
import copy
import logging
import gymnasium as gym
from .rl_config import SAC_HYPER_PARAMS_DEFAULT
from .env_config import REGISTRATION_DICT
from stable_baselines3 import A2C, PPO, DDPG, SAC
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from .env_config import BASE_CONFIG
from task.task1 import Navigation

logger = logging.getLogger(__name__)
def train(gym_env: str,
          total_timesteps: int,
          MODEL: base_class = PPO,
          model_save_path: str = "logs/PPO_docking",
          agent_hyper_params: dict = SAC_HYPER_PARAMS_DEFAULT,
          env_config: dict = BASE_CONFIG,
          tb_log_name: str = "SAC",
          timesteps_per_save: int = None,
          model_load_path: str = None,
          vector_env: int = None,
          tensorboard_log: str = None, 
          use_tensorboard: bool = True,
          eval_during_training: bool = False):
    
    if vector_env is not None:
        def make_env_(index):
            def _init():
                env_config_ = copy.deepcopy(env_config)
                env_config_["index"] = index
                env = make_gym(gym_env=gym_env, env_config=env_config_)
                env = Monitor(env)
                return env
            return _init

        envs = [make_env_(_) for _ in range(vector_env)]
        env = SubprocVecEnv(envs)
    else:
        env = make_gym(gym_env=gym_env, env_config=env_config)
    
    elapsed_timesteps = 0
    sim_timesteps = timesteps_per_save if timesteps_per_save else total_timesteps

    if use_tensorboard and tensorboard_log is None:
        tensorboard_log = os.path.join(os.path.dirname(model_save_path), "tensorboard")

    if model_load_path is None:
        model_kwargs = agent_hyper_params.copy()
        if use_tensorboard:
            model_kwargs['tensorboard_log'] = tensorboard_log

        valid_params = set(inspect.signature(SAC.__init__).parameters.keys()) - {'self'}
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}
        model = MODEL(policy='MlpPolicy', env=env, **filtered_kwargs) 
    else:
        model = MODEL.load(model_load_path, env=env)
        if use_tensorboard:
            model.tensorboard_log = tensorboard_log
        
    callbacks = []

    if use_tensorboard:
        training_callback = TrainingTensorboardCallback(verbose=1,log_interval=100000) 
        callbacks.append(training_callback)
    
    while elapsed_timesteps < total_timesteps:
        model.learn(total_timesteps=sim_timesteps, reset_num_timesteps=False, tb_log_name=tb_log_name)
        
        elapsed_timesteps = model.num_timesteps
        
        tmp_model_save_path = f"{model_save_path}_{elapsed_timesteps}"
        
        model.save(tmp_model_save_path)
        model.save_replay_buffer(tmp_model_save_path)
        logger.info(f'模型保存成功，保存路径为: {os.path.join(os.path.join(os.getcwd(), tmp_model_save_path))}')

    return None


class TrainingTensorboardCallback(BaseCallback):
    """
    自定义TensorBoard回调，记录训练过程中的额外指标
    """
    def __init__(self, verbose: int = 0, log_interval: int = 100):
        """
        初始化回调
        
        Args:
            verbose: 日志详细级别
            log_interval: 记录日志的间隔步数
        """
        super(TrainingTensorboardCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """
        在每个训练步骤执行
        """
        # 定期记录指标
        if self.n_calls % self.log_interval == 0:
            self._record_training_metrics()
        
        # 记录episode信息
        self._record_episode_info()
        
        return True
    
    def _record_training_metrics(self) -> None:
        """记录训练指标"""
        # 记录学习率
        if hasattr(self.model, 'lr_schedule'):
            try:
                lr = self.model.lr_schedule(self.model._current_progress_remaining)
                self.logger.record('train/learning_rate', lr)
            except:
                pass
        
        # 记录clip range（适用于PPO算法）
        if hasattr(self.model, 'clip_range'):
            try:
                clip_range = self.model.clip_range(self.model._current_progress_remaining)
                self.logger.record('train/clip_range', clip_range)
            except:
                pass
        
        # 记录熵系数（如果可用）
        if hasattr(self.model, 'ent_coef'):
            self.logger.record('train/entropy_coefficient', self.model.ent_coef)
    
    def _record_episode_info(self) -> None:
        """记录episode相关信息"""
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                if 'r' in ep_info:
                    self.episode_rewards.append(ep_info['r'])
                if 'l' in ep_info:
                    self.episode_lengths.append(ep_info['l'])
            
            # 计算并记录统计信息
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10:]  # 最近10个episode
                recent_lengths = self.episode_lengths[-10:]
                
                self.logger.record('train/mean_reward', np.mean(recent_rewards))
                self.logger.record('train/mean_ep_length', np.mean(recent_lengths))
                self.logger.record('train/max_reward', np.max(recent_rewards))
                self.logger.record('train/min_reward', np.min(recent_rewards))
                
                # 定期清理列表避免内存泄漏
                if len(self.episode_rewards) > 100:
                    self.episode_rewards = self.episode_rewards[-50:]
                    self.episode_lengths = self.episode_lengths[-50:]
    
    def _on_training_end(self) -> None:
        """训练结束时调用"""
        if self.verbose > 0:
            print("Training completed. TensorBoard logs have been saved.")

def make_gym(gym_env: str, env_config: dict):
    if gym_env in REGISTRATION_DICT:
        env = gym.make(gym_env, env_config=env_config)
        return env
    else:
        raise KeyError(f"此环境未定义或未进入config中,"
                       f" 当前可用环境为： {REGISTRATION_DICT.keys()}")