"""Training entrypoint for the Task4 obstacle-avoidance skill."""

from __future__ import annotations

import argparse
import copy
import inspect
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from smart_captain.simulation.defaults import DEFAULT_ENV_CONFIG
from smart_captain.skills.obstacle_avoidance.config import TASK4_SAC_HYPER_PARAMS


DEFAULT_ALGORITHM = "sac"
DEFAULT_TOTAL_TIMESTEPS = 6_000_000
DEFAULT_SAVE_INTERVAL = 200_000


def make_env(env_config: dict[str, Any] | None = None):
    """Create a Task4 obstacle-avoidance environment."""
    from smart_captain.skills.obstacle_avoidance.env import ObstacleAvoidanceEnv

    return ObstacleAvoidanceEnv(env_config=env_config or copy.deepcopy(DEFAULT_ENV_CONFIG))


def next_run_dir(base_dir: str | Path, prefix: str = "task4-v0_SAC") -> Path:
    """Return the next numbered run directory, e.g. `task4-v0_SAC_10`."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    run_nums: list[int] = []
    for directory in base_path.iterdir():
        if not directory.is_dir() or not directory.name.startswith(f"{prefix}_"):
            continue
        try:
            run_nums.append(int(directory.name.rsplit("_", 1)[1]))
        except ValueError:
            continue

    next_num = max(run_nums, default=0) + 1
    return base_path / f"{prefix}_{next_num}"


def _filter_model_kwargs(model_cls, hyper_params: dict[str, Any]) -> dict[str, Any]:
    valid_params = set(inspect.signature(model_cls.__init__).parameters) - {"self"}
    return {key: value for key, value in hyper_params.items() if key in valid_params}


def train(
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    timesteps_per_save: int = DEFAULT_SAVE_INTERVAL,
    env_config: dict[str, Any] | None = None,
    hyper_params: dict[str, Any] | None = None,
    run_dir: str | Path | None = None,
    model_load_path: str | Path | None = None,
    tb_log_name: str | None = None,
    progress_bar: bool = False,
):
    """Train Task4 obstacle avoidance with SAC and periodic checkpoint saves."""
    from stable_baselines3 import SAC

    used_config = copy.deepcopy(env_config or DEFAULT_ENV_CONFIG)
    output_dir = Path(run_dir) if run_dir is not None else next_run_dir("logs1")
    output_dir.mkdir(parents=True, exist_ok=True)
    used_config["save_path_folder"] = str(output_dir)

    env = make_env(used_config)
    tensorboard_log = output_dir / "tensorboard"
    params = copy.deepcopy(hyper_params or TASK4_SAC_HYPER_PARAMS)
    params["tensorboard_log"] = str(tensorboard_log)

    if model_load_path is None:
        model = SAC("MlpPolicy", env, **_filter_model_kwargs(SAC, params))
    else:
        model = SAC.load(str(model_load_path), env=env)
        model.tensorboard_log = str(tensorboard_log)

    elapsed_timesteps = 0
    checkpoint_base = output_dir / "task4-v0_SAC"
    tb_name = tb_log_name or output_dir.name

    while elapsed_timesteps < total_timesteps:
        learn_steps = min(timesteps_per_save, total_timesteps - elapsed_timesteps)
        model.learn(
            total_timesteps=learn_steps,
            reset_num_timesteps=False,
            tb_log_name=tb_name,
            progress_bar=progress_bar,
        )
        elapsed_timesteps = model.num_timesteps
        checkpoint_path = f"{checkpoint_base}_{elapsed_timesteps}"
        model.save(checkpoint_path)
        model.save_replay_buffer(checkpoint_path)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Task4 obstacle-avoidance skill.")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--model-load-path", default=None)
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()

    train(
        total_timesteps=args.total_timesteps,
        timesteps_per_save=args.save_interval,
        run_dir=args.run_dir,
        model_load_path=args.model_load_path,
        progress_bar=args.progress_bar,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_ALGORITHM",
    "DEFAULT_SAVE_INTERVAL",
    "DEFAULT_TOTAL_TIMESTEPS",
    "make_env",
    "next_run_dir",
    "train",
]
