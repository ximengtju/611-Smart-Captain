import copy
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.rl_config import PPO_HYPER_PARAMS_TEST, SAC_HYPER_PARAMS_TEST
from stable_baselines3 import A2C, PPO, DDPG, SAC
from env.env_config import BASE_CONFIG
import env.trainer as train
import matplotlib as mpl
import os

mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12

GYM_ENV = ["task2-v0"]

MODELS = [
    SAC,
]
MODELS_STR = [
    "_SAC",
]

HYPER_PARAMS = [
    SAC_HYPER_PARAMS_TEST,
]

if __name__ == "__main__":
    for K, MODEL in enumerate(MODELS):
        """这里是为了，可以一次性进行多个算法，多个不同任务的训练，但是一般情况下建议分别训练"""    
        used_TRAIN_CONFIG = copy.deepcopy(BASE_CONFIG)
        for GYM in GYM_ENV:
            log_dir = os.path.join(os.getcwd(), "logs2/")
            log_dir = Path(log_dir)
            file_name_prefix = GYM + MODELS_STR[K]
            exst_run_nums = [int(str(folder.name).split(file_name_prefix)[1].split("_")[1]) for folder in
                            log_dir.iterdir() if
                            str(folder.name).startswith(file_name_prefix)]
            if len(exst_run_nums) == 0:
                urr_run = file_name_prefix + "_" + '1'
            else:
                urr_run = file_name_prefix + "_" + '%i' % (max(exst_run_nums) + 1)
            
            used_TRAIN_CONFIG["save_path_folder"] = os.path.join(os.getcwd(), "logs2/", urr_run)

            train.train(gym_env=GYM,
                        total_timesteps=10000000,
                        MODEL=MODEL,
                        model_save_path="logs2/" + urr_run + "/" + GYM + MODELS_STR[K],
                        tb_log_name=urr_run,

                        agent_hyper_params=HYPER_PARAMS[K],
                        env_config=used_TRAIN_CONFIG,
                        timesteps_per_save=100000,
                        model_load_path=None,
                        vector_env=2, )