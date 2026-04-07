import numpy as np
import gymnasium as gym
from env.env_config import BASE_CONFIG
from env.agents import Agents
from task.task_combine import MAInterface
from llm.Interface import LLMInterface  
from stable_baselines3 import SAC, A2C, PPO

# 当前的任务分解，方便起见，使用的仍是去年九月份的思路，大模型一次性分解完，然后完全丢给RL去执行，实际上这种做法不可行
# 但修改方案仍在探讨

auv = MAInterface(env_config=BASE_CONFIG, mode=0)
obs, info = auv.reset()


model_paths = [
    'logs1/task1-v0_SAC_3/task1-v0_SAC_400000.zip',   # 模式 0（导航）
    'logs2/task2-v0_SAC_1/task2-v0_SAC_200000.zip'    # 模式 1（目标跟踪）
]
model_class_dict = {'sac': SAC, 'a2c': A2C, 'ppo': PPO}
model_types = ['sac', 'sac']
agents = Agents(
    model_paths=model_paths,
    model_types=model_types,
    env=auv,
    model_class_dict=model_class_dict,
    mode='multi'
)


llm_model_path = "/path/to/your/llm_model"
llm_interface = LLMInterface(
    llm_path=llm_model_path,
    env=auv,  
    agents=agents,   
)


main_task = "请控制水下机器人躲避障碍物并跟踪一个移动目标"
mode_indices = llm_interface.set_mode_from_task(main_task)
current_subtask_idx = 0
total_subtasks = len(mode_indices)
print(f"共有 {total_subtasks} 个子任务，模式序列: {mode_indices}")


obs, info = auv.reset()
while current_subtask_idx < total_subtasks:
    done = False
    while not done:
        action, _ = agents.predict(obs)
        obs, reward, done, truncated, info = auv.step(action)
    
    current_subtask_idx += 1
    if current_subtask_idx < total_subtasks:
        next_mode = mode_indices[current_subtask_idx]
        auv.set_multi_mode_index(next_mode)
        agents.set_multi_mode_index(next_mode)
        obs, info = auv.reset()
    else:
        print("\n所有子任务已完成！")