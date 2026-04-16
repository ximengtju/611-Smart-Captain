"""Scenario loading helpers for the new simulation layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScenarioSpec:
    """Serializable scenario descriptor."""

    name: str
    world: str
    package_name: str
    main_agent: str
    ticks_per_sec: int
    agents: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a HoloOcean-compatible scenario dictionary."""
        return {
            "name": self.name,
            "world": self.world,
            "package_name": self.package_name,
            "main_agent": self.main_agent,
            "ticks_per_sec": self.ticks_per_sec,
            "agents": self.agents,
        }
