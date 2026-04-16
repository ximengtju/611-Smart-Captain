"""Training configuration helpers for the restructured RL layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingRequest:
    """Describe a training job for one skill."""

    skill: str
    algorithm: str
    total_timesteps: int
    env_config: dict[str, Any] = field(default_factory=dict)
    hyper_params: dict[str, Any] = field(default_factory=dict)
