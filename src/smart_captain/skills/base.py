"""Shared abstractions for RL-executable skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillSpec:
    """Declarative description of a low-level executable skill."""

    name: str
    env_cls: str
    policy_cls: str
    default_scenario: str
    observation_dim: int
    action_dim: int
    description: str
    default_sensors: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    train_entrypoint: str | None = None
    reward_entrypoint: str | None = None
    config_entrypoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillAdapter:
    """Base class for future migrated skill environments."""

    spec: SkillSpec

    def __init__(self, runtime_env: Any | None = None) -> None:
        self.runtime_env = runtime_env

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def step(self, action: Any) -> Any:
        raise NotImplementedError
