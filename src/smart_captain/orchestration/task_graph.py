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
    # Recent environment snapshots are kept at mission scope so a future
    # replanner can explain where the failure happened, not just which skill
    # failed.  Records are compact dictionaries rather than raw observations.
    feedback_history: list[dict[str, Any]] = field(default_factory=list)
    # Each evaluator writes its latest result here under the subtask id.  This
    # separates progress judgment from raw shared world facts.
    progress_by_subtask: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Retry accounting belongs to the mission context because retry policy must
    # survive multiple dispatcher calls for the same active subtask.
    retry_counts: dict[str, int] = field(default_factory=dict)
    # These fields are the handoff point to future replanning.  The first
    # implementation only marks the request; a later replanner can consume the
    # graph, world_state, feedback_history, and failed_subtasks together.
    replan_requested: bool = False
    replan_reason: str | None = None
    # Decision records make the closed-loop behavior inspectable after a run.
    decisions: list[dict[str, Any]] = field(default_factory=list)
