"""Unified mission runner for the restructured Smart Captain framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smart_captain.app.compat_runtime import (
    LegacyRuntimeConfig,
    create_adapter_backed_legacy_runtime,
    execute_adapter_backed_legacy_plan,
    preview_legacy_plan,
)
from smart_captain.llm.interface import StructuredLLMInterface
from smart_captain.orchestration.dispatcher import TaskDispatcher
from smart_captain.orchestration.task_graph import TaskGraph
from smart_captain.rl.model_store import ModelStore


@dataclass(frozen=True)
class MissionRequest:
    """Input request for the new mission runner."""

    mission_id: str
    command: str
    world_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionRunner:
    """High-level entrypoint that ties planning, orchestration, and runtime prep."""

    llm_interface: StructuredLLMInterface = field(default_factory=StructuredLLMInterface)
    dispatcher: TaskDispatcher = field(default_factory=TaskDispatcher)
    model_store: ModelStore = field(default_factory=ModelStore)

    def plan(self, request: MissionRequest) -> TaskGraph:
        """Build a task graph from a mission request."""
        return self.llm_interface.parse_command(
            command=request.command,
            mission_id=request.mission_id,
            world_state=request.world_state,
        )

    def plan_summary(self, request: MissionRequest) -> dict[str, Any]:
        """Return a serializable plan-only summary."""
        graph = self.plan(request)
        context = self.dispatcher.bootstrap_context(graph, initial_world_state=request.world_state)
        current = self.dispatcher.step(graph, context)
        return {
            "mission_id": graph.mission_id,
            "command": request.command,
            "llm_backend": self.llm_interface.get_last_source(),
            "skills": [subtask.skill for subtask in graph.subtasks],
            "subtask_ids": [subtask.id for subtask in graph.subtasks],
            "scenarios": [subtask.scenario for subtask in graph.subtasks],
            "sensor_modes": [subtask.sensor_mode for subtask in graph.subtasks],
            "first_active_skill": current.skill if current else None,
            "model_store": self.model_store.to_summary(),
        }

    def legacy_preview(self, request: MissionRequest) -> dict[str, Any]:
        """Return a summary of how the mission would bind to the compatibility runtime."""
        config = LegacyRuntimeConfig(
            mission_id=request.mission_id,
            command=request.command,
            world_state=request.world_state,
        )
        return preview_legacy_plan(config)

    def create_legacy_adapter_runtime(self, request: MissionRequest):
        """Build the adapter-backed compatibility runtime for a mission.

        This is the first new-framework entrypoint that can compose the shared
        AUV adapter instead of going directly through `task_combine1.py`.
        """
        config = LegacyRuntimeConfig(
            mission_id=request.mission_id,
            command=request.command,
            world_state=request.world_state,
        )
        return create_adapter_backed_legacy_runtime(config)

    def execute(self, request: MissionRequest) -> dict[str, Any]:
        """Execute a mission using the adapter-backed compatibility runtime."""
        config = LegacyRuntimeConfig(
            mission_id=request.mission_id,
            command=request.command,
            world_state=request.world_state,
        )
        return execute_adapter_backed_legacy_plan(config)
