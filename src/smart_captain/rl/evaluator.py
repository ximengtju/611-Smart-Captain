"""Evaluation helpers for the restructured RL layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationSummary:
    """Serializable evaluation result."""

    skill: str
    episode_count: int
    mean_reward: float
    metadata: dict[str, Any]
