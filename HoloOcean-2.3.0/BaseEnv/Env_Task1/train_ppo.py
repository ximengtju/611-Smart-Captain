import copy
from pathlib import Path
import os
import torch
import matplotlib as mpl

from stable_baselines3 import PPO
from env.rl_config import PPO_HYPER_PARAMS_TEST
from env.env_config import BASE_CONFIG
from env.trainer_ppo import train

print("cuda可用:", torch.cuda.is_available())

mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12

GYM_ENV = ["task1-v0"]

MODELS = [PPO]
MODELS_STR = ["_PPO"]
HYPER_PARAMS = [PPO_HYPER_PARAMS_TEST]

if __name__ == "__main__":
    for k, model_cls in enumerate(MODELS):
        used_train_config = copy.deepcopy(BASE_CONFIG)

        for gym_env in GYM_ENV:
            log_dir = Path(os.path.join(os.getcwd(), "logs1"))
            log_dir.mkdir(parents=True, exist_ok=True)

            file_name_prefix = gym_env + MODELS_STR[k]
            exst_run_nums = [
                int(str(folder.name).split(file_name_prefix)[1].split("_")[1])
                for folder in log_dir.iterdir()
                if str(folder.name).startswith(file_name_prefix)
            ]

            if len(exst_run_nums) == 0:
                curr_run = file_name_prefix + "_1"
            else:
                curr_run = file_name_prefix + "_" + str(max(exst_run_nums) + 1)

            used_train_config["save_path_folder"] = os.path.join(
                os.getcwd(), "logs1", curr_run
            )

            train(
                gym_env=gym_env,
                total_timesteps=1500000,
                MODEL=model_cls,
                model_save_path=os.path.join("logs1", curr_run, gym_env + MODELS_STR[k]),
                tb_log_name=curr_run,
                agent_hyper_params=HYPER_PARAMS[k],
                env_config=used_train_config,
                timesteps_per_save=100000,
                model_load_path=None,
                vector_env=None,
            )