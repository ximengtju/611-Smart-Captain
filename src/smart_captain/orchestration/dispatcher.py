"""Deterministic execution skeleton for task orchestration.

This module is the migration target for the hard-coded switching logic in
`Main-Framework/task/task_combine*.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smart_captain.orchestration.registry import (
    resolve_scenario_class,
    resolve_skill_env_class,
    resolve_skill_spec,
)
from smart_captain.orchestration.task_graph import ExecutionContext, Subtask, TaskGraph, TaskStatus


@dataclass
class ActivationRecord:
    """Resolved runtime selection for one subtask."""

    subtask_id: str
    skill: str
    scenario: str
    sensor_mode: str | None
    env_cls: type[Any]
    scenario_cls: type[Any]


@dataclass
class TaskDispatcher:
    """Dispatch structured subtasks to registered skill adapters."""

    activation_history: list[ActivationRecord] = field(default_factory=list)

    def activate(self, subtask: Subtask, context: ExecutionContext) -> ActivationRecord:
        """Resolve all runtime components needed for a subtask."""
        spec = resolve_skill_spec(subtask.skill)
        scenario_name = subtask.scenario or spec.default_scenario
        env_cls = resolve_skill_env_class(subtask.skill)
        scenario_cls = resolve_scenario_class(scenario_name)

        context.active_skill = subtask.skill
        context.active_scenario = scenario_name
        context.active_sensor_mode = subtask.sensor_mode
        subtask.status = TaskStatus.ACTIVE

        record = ActivationRecord(
            subtask_id=subtask.id,
            skill=subtask.skill,
            scenario=scenario_name,
            sensor_mode=subtask.sensor_mode,
            env_cls=env_cls,
            scenario_cls=scenario_cls,
        )
        self.activation_history.append(record)
        return record

    def mark_succeeded(self, subtask: Subtask, context: ExecutionContext) -> None:
        """Mark the currently active subtask as succeeded."""
        subtask.status = TaskStatus.SUCCEEDED
        if subtask.id not in context.completed_subtasks:
            context.completed_subtasks.append(subtask.id)

    def mark_failed(self, subtask: Subtask, context: ExecutionContext) -> None:
        """Mark the currently active subtask as failed."""
        subtask.status = TaskStatus.FAILED
        if subtask.id not in context.failed_subtasks:
            context.failed_subtasks.append(subtask.id)

    def next_pending(self, graph: TaskGraph) -> Subtask | None:
        """Return the next unfinished subtask."""
        return graph.current_subtask()

    def bootstrap_context(
        self,
        graph: TaskGraph,
        initial_world_state: dict[str, Any] | None = None,
    ) -> ExecutionContext:
        """Create a fresh execution context for one mission."""
        return ExecutionContext(
            mission_id=graph.mission_id,
            world_state=initial_world_state or {},
        )

    def update_world_state(
        self,
        context: ExecutionContext,
        new_state: dict[str, Any],
    ) -> ExecutionContext:
        """Merge new world-state values into the current context."""
        context.world_state.update(new_state)
        return context

    def step(
        self,
        graph: TaskGraph,
        context: ExecutionContext,
        outcome: str | None = None,
        world_state_update: dict[str, Any] | None = None,
    ) -> Subtask | None:
        """Advance the task graph by applying the latest subtask outcome.

        Supported outcomes:
        - `None`: only activate the next pending task
        - `"succeeded"`: mark current task as done and advance
        - `"failed"`: mark current task as failed and advance
        """
        current = graph.current_subtask()
        if current is None:
            return None

        if current.status == TaskStatus.PENDING:
            self.activate(current, context)

        if world_state_update:
            self.update_world_state(context, world_state_update)

        if outcome == "succeeded":
            self.mark_succeeded(current, context)
        elif outcome == "failed":
            self.mark_failed(current, context)

        next_task = graph.current_subtask()
        if next_task is not None and next_task.status == TaskStatus.PENDING:
            self.activate(next_task, context)
        return next_task
