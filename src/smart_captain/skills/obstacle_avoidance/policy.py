"""Policy wrapper for the Task4 obstacle-avoidance skill."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_captain.rl.model_store import ModelStore


@dataclass
class ObstacleAvoidancePolicy:
    """Lazy-loading wrapper around a Stable Baselines3 policy."""

    model_path: str | Path | None = None
    algorithm: str = "sac"
    device: str | None = None
    model: Any | None = None

    def resolve_model_path(self) -> Path:
        """Return the configured model path or the registered default path."""
        if self.model_path is not None:
            return Path(self.model_path)
        return ModelStore().get("obstacle_avoidance").absolute_path

    def load(self):
        """Load and cache the obstacle-avoidance policy."""
        if self.model is not None:
            return self.model
        if self.algorithm != "sac":
            raise ValueError(f"Unsupported obstacle-avoidance algorithm: {self.algorithm}")

        from stable_baselines3 import SAC

        kwargs = {}
        if self.device is not None:
            kwargs["device"] = self.device
        self.model = SAC.load(str(self.resolve_model_path()), **kwargs)
        return self.model

    def predict(self, observation, state=None, deterministic: bool = True):
        """Predict an action using the loaded policy."""
        model = self.load()
        return model.predict(observation, state=state, deterministic=deterministic)


__all__ = ["ObstacleAvoidancePolicy"]
