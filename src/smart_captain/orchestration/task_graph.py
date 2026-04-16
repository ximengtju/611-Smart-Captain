"""Structured task graph types for the orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Execution status for an individual subtask."""

    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    """Structured executable subtask.

    This is the stable interface between the LLM planner output and the
    deterministic task orchestrator.
    """

    id: str
    skill: str
    params: dict[str, Any] = field(default_factory=dict)
    scenario: str | None = None
    sensor_mode: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    success_condition: str | None = None
    failure_condition: str | None = None
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class TaskGraph:
    """Linear task graph for the current migration phase.

    This keeps the first implementation simple while preserving a structure that
    can later grow into DAGs or branching graphs.
    """

    mission_id: str
    subtasks: list[Subtask]
    mission_metadata: dict[str, Any] = field(default_factory=dict)

    def current_index(self) -> int | None:
        """Return the first unfinished subtask index."""
        for index, subtask in enumerate(self.subtasks):
            if subtask.status in {TaskStatus.PENDING, TaskStatus.ACTIVE}:
                return index
        return None

    def current_subtask(self) -> Subtask | None:
        """Return the first unfinished subtask, if any."""
        index = self.current_index()
        return None if index is None else self.subtasks[index]

    def is_complete(self) -> bool:
        """Return whether all subtasks reached terminal states."""
        return all(
            subtask.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.SKIPPED}
            for subtask in self.subtasks
        )


@dataclass
class ExecutionContext:
    """Mutable orchestration context shared during mission execution."""

    mission_id: str
    world_state: dict[str, Any] = field(default_factory=dict)
    active_skill: str | None = None
    active_scenario: str | None = None
    active_sensor_mode: str | None = None
    completed_subtasks: list[str] = field(default_factory=list)
    failed_subtasks: list[str] = field(default_factory=list)
