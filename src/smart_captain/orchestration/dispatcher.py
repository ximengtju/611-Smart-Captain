"""Deterministic execution skeleton for task orchestration.

This module is the migration target for the hard-coded switching logic in
`Main-Framework/task/task_combine*.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smart_captain.orchestration.feedback import FeedbackSnapshot, TaskProgress
from smart_captain.orchestration.registry import (
    resolve_scenario_class,
    resolve_skill_env_class,
    resolve_skill_spec,
)
from smart_captain.orchestration.recovery import (
    DecisionAction,
    OrchestrationDecision,
    RecoveryPolicy,
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
    recovery_policy: RecoveryPolicy = field(default_factory=RecoveryPolicy)

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
        """Merge new world-state values into the current context.

        Closed-loop feedback uses nested namespaces such as `vehicle`,
        `environment`, and `skills.navigation`.  A shallow `dict.update()` would
        overwrite an entire namespace whenever one skill writes a new metric, so
        this method performs a recursive merge for dictionaries and normal
        replacement for scalar values.
        """
        self._merge_nested_dict(context.world_state, new_state)
        return context

    def record_feedback(
        self,
        context: ExecutionContext,
        feedback: FeedbackSnapshot,
    ) -> None:
        """Store feedback history and expose snapshot facts in world_state."""
        context.feedback_history.append(feedback.to_history_record())
        self.update_world_state(context, feedback.to_world_state_delta())

    def record_progress(
        self,
        context: ExecutionContext,
        progress: TaskProgress,
    ) -> None:
        """Store evaluator output and merge evaluator state deltas."""
        context.progress_by_subtask[progress.subtask_id] = progress.to_record()
        if progress.world_state_delta:
            self.update_world_state(context, progress.world_state_delta)

    @staticmethod
    def _merge_nested_dict(target: dict[str, Any], update: dict[str, Any]) -> None:
        """Recursively merge `update` into `target` in place.

        This keeps shared state stable across skills.  For example, an
        obstacle-avoidance update to `skills.obstacle_avoidance` will not delete
        the latest `skills.navigation` values.
        """
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                TaskDispatcher._merge_nested_dict(target[key], value)
            else:
                target[key] = value

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

    def step_with_progress(
        self,
        graph: TaskGraph,
        context: ExecutionContext,
        progress: TaskProgress,
        feedback: FeedbackSnapshot | None = None,
        record_history: bool = True,
    ) -> OrchestrationDecision:
        """Advance orchestration from evaluator output.

        This is the feedback-driven companion to the older `step()` method.
        The old method is left intact for plan previews and manual outcome
        injection; runtime loops can call this method when they have real
        environment feedback and a skill evaluator result.
        """
        current = graph.current_subtask()
        if current is None:
            return OrchestrationDecision(
                action=DecisionAction.ABORT,
                subtask_id=progress.subtask_id,
                skill=progress.skill,
                reason="no_active_subtask",
                progress_status=progress.status,
            )

        if current.status == TaskStatus.PENDING:
            self.activate(current, context)

        if feedback is not None:
            # Always refresh the shared `world_state` so evaluators and future
            # replanners see the latest vehicle/environment facts.  History is
            # sampled by the runtime to keep JSON artifacts small.
            self.update_world_state(context, feedback.to_world_state_delta())
            if record_history:
                context.feedback_history.append(feedback.to_history_record())
        self.record_progress(context, progress)

        decision = self.recovery_policy.decide(
            progress=progress,
            subtask=current,
            context=context,
        )
        if record_history or decision.action != DecisionAction.CONTINUE_CURRENT:
            # Ordinary "keep going" decisions can happen tens of thousands of
            # times.  We sample them, but always keep meaningful events such as
            # advance, retry, replan, and abort.
            context.decisions.append(decision.to_record())

        if decision.action == DecisionAction.ADVANCE:
            self.mark_succeeded(current, context)
            next_task = graph.current_subtask()
            if next_task is not None and next_task.status == TaskStatus.PENDING:
                self.activate(next_task, context)
            return decision

        if decision.action == DecisionAction.RETRY:
            # The runtime is responsible for deciding whether a retry means
            # continuing the same episode or resetting a skill env.  The
            # dispatcher only records that recovery policy consumed one retry.
            context.retry_counts[current.id] = decision.retry_count
            return decision

        if decision.action == DecisionAction.SWITCH_SKILL:
            return decision

        if decision.action in {DecisionAction.REQUEST_REPLAN, DecisionAction.ABORT}:
            self.mark_failed(current, context)
            context.replan_requested = decision.action == DecisionAction.REQUEST_REPLAN
            context.replan_reason = decision.reason
            return decision

        return decision
