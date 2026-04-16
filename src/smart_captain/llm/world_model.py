"""Domain knowledge and world-state hints used by the LLM layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityProfile:
    """Capabilities that the planner can assume are available."""

    available_skills: tuple[str, ...]
    available_sensors: tuple[str, ...]
    available_scenarios: tuple[str, ...]


@dataclass
class WorldModel:
    """Compact representation of knowledge exposed to the planner layer."""

    capability_profile: CapabilityProfile
    environment_knowledge: dict[str, Any] = field(default_factory=dict)
    mission_protocols: dict[str, Any] = field(default_factory=dict)
    vehicle_constraints: dict[str, Any] = field(default_factory=dict)

    def build_prompt_context(self, world_state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a structured context bundle for future model prompting."""
        return {
            "available_skills": self.capability_profile.available_skills,
            "available_sensors": self.capability_profile.available_sensors,
            "available_scenarios": self.capability_profile.available_scenarios,
            "environment_knowledge": self.environment_knowledge,
            "mission_protocols": self.mission_protocols,
            "vehicle_constraints": self.vehicle_constraints,
            "world_state": world_state or {},
        }
