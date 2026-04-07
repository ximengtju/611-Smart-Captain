from env.pierharbor_hovering import BaseEnvironment
from env.pierharbor_hovering import BASE_CONFIG
import numpy as np

class Navigation(BaseEnvironment):
    def __init__(self, env_config: dict = BASE_CONFIG, auv = None, train_mode = True):
        super().__init__(env_config, auv, train_mode)
        self.r_min = env_config["r_min"]
        self.r_max = env_config["r_max"]
        

    def generate_environment(self, manual_import: bool = False, import_location = None):
        super().generate_environment()

        if manual_import == False:
            psi_auv = self.auv_attitude[2]
            r_target = np.random.uniform(self.r_min, self.r_max)
            psi_target_rel = np.random.uniform(-np.pi, np.pi) 
            psi_target_abs = self.ssa(psi_auv + psi_target_rel) 

            x_target = r_target * np.cos(psi_target_abs)
            y_target = r_target * np.sin(psi_target_abs)
            z_target = np.random.uniform(-15.0, -5.0)
            self.goal_location = np.array([x_target, y_target, z_target])
        else:
            self.goal_location = import_location
    