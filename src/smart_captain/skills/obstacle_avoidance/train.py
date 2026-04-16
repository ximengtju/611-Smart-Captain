"""Obstacle-avoidance training config bridge."""

from __future__ import annotations


DEFAULT_ALGORITHM = "sac"

DEFAULT_HYPER_PARAMS = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 100,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "tensorboard_log": None,
    "create_eval_env": False,
    "policy_kwargs": None,
    "verbose": 0,
    "seed": None,
    "device": "cuda:0",
    "_init_setup_model": True,
}
