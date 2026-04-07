from stable_baselines3 import SAC, A2C
import gymnasium as gym
from env.env_config import BASE_CONFIG
from env.agents import Agents
from task.task_combine1 import MAInterface

# auv = gym.make("task2-v0", env_config=BASE_CONFIG)
auv = MAInterface(env_config=BASE_CONFIG)
obs, info = auv.reset()
model_class_dict = {'sac': SAC, 'a2c': A2C}
model_path = ['logs1/task1-v0_SAC_3/task1-v0_SAC_400000.zip',
              'logs2/task2-v0_SAC_1/task2-v0_SAC_200000.zip']
model_type = ['sac', 'sac']
agents = Agents(model_paths=model_path, model_types=model_type, env=auv, 
                model_class_dict=model_class_dict, mode='multi')
agents.set_multi_mode_index(0)
for i in range(200):
    if i == 100:
        auv.set_multi_mode_index(1)
    action, _ = agents.predict(obs)
    obs, reward, done, _, info = auv.step(action)
    if i == 100:
        agents.set_multi_mode_index(1)