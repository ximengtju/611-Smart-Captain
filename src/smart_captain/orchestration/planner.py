"""Task-graph construction helpers for the orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_captain.orchestration.task_graph import Subtask, TaskGraph
from smart_captain.orchestration.registry import resolve_skill_spec


@dataclass
class TaskGraphPlanner:
    """Build a deterministic task graph from structured planner output."""

    default_sensor_mode: str | None = None

    def build_subtask(self, index: int, payload: dict[str, Any]) -> Subtask:
        """Convert a structured payload into a validated `Subtask`."""
        skill_name = payload["skill"]
        spec = resolve_skill_spec(skill_name)
        return Subtask(
            id=payload.get("id", f"task_{index}"),
            skill=skill_name,
            params=payload.get("params", {}),
            scenario=payload.get("scenario", spec.default_scenario),
            sensor_mode=payload.get("sensor_mode", self.default_sensor_mode),
            constraints=payload.get("constraints", {}),
            success_condition=payload.get("success_condition"),
            failure_condition=payload.get("failure_condition"),
            description=payload.get("description", spec.description),
        )

    def build_graph(
        self,
        mission_id: str,
        structured_tasks: list[dict[str, Any]],
        mission_metadata: dict[str, Any] | None = None,
    ) -> TaskGraph:
        """Build a linear task graph from a sequence of structured subtasks."""
        subtasks = [
            self.build_subtask(index=index, payload=payload)
            for index, payload in enumerate(structured_tasks)
        ]
        return TaskGraph(
            mission_id=mission_id,
            subtasks=subtasks,
            mission_metadata=mission_metadata or {},
        )
