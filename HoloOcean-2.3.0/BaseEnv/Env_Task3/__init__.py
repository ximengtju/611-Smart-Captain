from gymnasium.envs.registration import register
from env.env_config import REGISTRATION_DICT

#把配置表里的环境名字都注册成 Gym 环境
for ide, entry_p in REGISTRATION_DICT.items():
    register(
        id=ide,
        entry_point=entry_p
    )