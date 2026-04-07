from env.trainer import make_gym
import torch

class Evaluator:
    def __init__(self,
                 model_path: str,
                 env_name: str,
                 model_type: str = "SAC",
                 output_dir: str = "./visualizations",
                 seed: int = 42):
        self.model_path = model_path
        self.env_name = env_name
        self.model_type = model_type
        self.output_dir = output_dir
        self.seed = seed

        