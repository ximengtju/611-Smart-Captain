"""Unified LLM-facing interface for structured mission planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smart_captain.llm.decomposer import StructuredTaskDecomposer
from smart_captain.llm.world_model import CapabilityProfile, WorldModel
from smart_captain.orchestration.planner import TaskGraphPlanner
from smart_captain.orchestration.task_graph import TaskGraph
from smart_captain.simulation.registry import SCENARIO_REGISTRY, SENSOR_REGISTRY
from smart_captain.skills.registry import SKILL_REGISTRY


@dataclass
class StructuredLLMInterface:
    """Bridge natural-language mission input to orchestration-ready task graphs."""

    world_model: WorldModel = field(default_factory=lambda: WorldModel(
        capability_profile=CapabilityProfile(
            available_skills=tuple(SKILL_REGISTRY.keys()),
            available_sensors=tuple(SENSOR_REGISTRY.keys()),
            available_scenarios=tuple(SCENARIO_REGISTRY.keys()),
        ),
        mission_protocols={
            "task_decomposition": "High-level commands should be mapped to executable RL skills.",
            "sensor_switching": "Use imaging sonar or camera only when semantically required.",
        },
    ))
    decomposer: StructuredTaskDecomposer = field(default_factory=StructuredTaskDecomposer)
    planner: TaskGraphPlanner = field(default_factory=TaskGraphPlanner)
    last_payloads: list[dict[str, Any]] | None = None
    last_graph: TaskGraph | None = None

    def parse_command(
        self,
        command: str,
        mission_id: str = "mission_0",
        world_state: dict[str, Any] | None = None,
    ) -> TaskGraph:
        """Convert a natural-language command into a validated `TaskGraph`."""
        world_context = self.world_model.build_prompt_context(world_state=world_state)
        payloads = self.decomposer.decompose(command=command, world_context=world_context)
        graph = self.planner.build_graph(
            mission_id=mission_id,
            structured_tasks=payloads,
            mission_metadata={"command": command},
        )
        self.last_payloads = payloads
        self.last_graph = graph
        return graph

    def get_last_payloads(self) -> list[dict[str, Any]] | None:
        """Return the latest structured payload sequence."""
        return self.last_payloads

    def get_last_source(self) -> str:
        """Return which decomposition backend handled the last command."""
        return self.decomposer.last_source

    def get_last_graph(self) -> TaskGraph | None:
        """Return the latest built task graph."""
        return self.last_graph
