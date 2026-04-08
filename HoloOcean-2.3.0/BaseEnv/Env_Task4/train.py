import copy
from pathlib import Path
import torch.cuda

from BaseEnv.Env_Task4.env.rl_config import SAC_HYPER_PARAMS_TEST
from stable_baselines3 import SAC
from BaseEnv.Env_Task4.env.env_config import BASE_CONFIG
from BaseEnv.Env_Task4.env import trainer as train
import matplotlib as mpl
import os

print("cuda可用:",torch.cuda.is_available())

mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12

GYM_ENV = ["task4-v0"]

MODELS = [
    SAC,
]
MODELS_STR = [
    "_SAC",
]
#超参数
HYPER_PARAMS = [
    SAC_HYPER_PARAMS_TEST,
]

if __name__ == "__main__":
    for K, MODEL in enumerate(MODELS):
        used_TRAIN_CONFIG = copy.deepcopy(BASE_CONFIG)
        for GYM in GYM_ENV:
            #生成日志
            log_dir = os.path.join(os.getcwd(), "logs1/")
            log_dir = Path(log_dir)
            file_name_prefix = GYM + MODELS_STR[K]
            log_dir.mkdir(parents=True, exist_ok=True)
            exst_run_nums = [int(str(folder.name).split(file_name_prefix)[1].split("_")[1]) for folder in
                            log_dir.iterdir() if
                            str(folder.name).startswith(file_name_prefix)]
            if len(exst_run_nums) == 0:
                urr_run = file_name_prefix + "_" + '1'
            else:
                urr_run = file_name_prefix + "_" + '%i' % (max(exst_run_nums) + 1)
            
            used_TRAIN_CONFIG["save_path_folder"] = os.path.join(os.getcwd(), "logs1/", urr_run)

            #训练主函数
            train.train(gym_env=GYM,
                        total_timesteps=2000000,
                        MODEL=MODEL,
                        model_save_path="logs1/" + urr_run + "/" + GYM + MODELS_STR[K],
                        tb_log_name=urr_run,

                        agent_hyper_params=HYPER_PARAMS[K],
                        env_config=used_TRAIN_CONFIG,
                        timesteps_per_save=100000,
                        model_load_path=None,
                        vector_env=None, )