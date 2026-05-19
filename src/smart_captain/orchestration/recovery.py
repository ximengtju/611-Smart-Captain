"""Recovery decisions for progress-driven orchestration.

This layer is deliberately separate from skill evaluators.  Evaluators explain
the active skill's condition; recovery policy decides whether that condition
should continue, advance, retry locally, request replanning, or abort.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from smart_captain.orchestration.feedback import ProgressStatus, TaskProgress
from smart_captain.orchestration.task_graph import ExecutionContext, Subtask


# 重规划相关的决策
class DecisionAction(str, Enum):
    """Dispatcher-level action selected after evaluating task progress."""

    CONTINUE_CURRENT = "continue_current"
    ADVANCE = "advance"
    RETRY = "retry"
    SWITCH_SKILL = "switch_skill"
    REQUEST_REPLAN = "request_replan"
    ABORT = "abort"


@dataclass(frozen=True)
class OrchestrationDecision:
    """Decision returned by the dispatcher after applying recovery policy."""

    action: DecisionAction
    subtask_id: str
    skill: str
    reason: str
    progress_status: ProgressStatus
    retry_count: int = 0
    details: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        """Return a compact record for logs and future replan requests."""
        return {
            "action": self.action.value,
            "subtask_id": self.subtask_id,
            "skill": self.skill,
            "reason": self.reason,
            "progress_status": self.progress_status.value,
            "retry_count": self.retry_count,
            "details": self.details or {},
        }


@dataclass
class RecoveryPolicy:
    """Small first-pass recovery policy with a replan-ready interface.

    The first implementation keeps behavior conservative: successful progress
    advances, normal progress continues, blocked progress can retry a limited
    number of times, and terminal failures request replanning.  A future LLM or
    rule-based replanner can consume the decision records without changing the
    evaluator API.
    """

    max_blocked_retries: int = 1

    def decide(
        self,
        *,
        progress: TaskProgress,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> OrchestrationDecision:
        """Choose the orchestration action for one progress update."""
        retry_count = context.retry_counts.get(subtask.id, 0)

        if progress.status == ProgressStatus.SUCCEEDED:
            return OrchestrationDecision(
                action=DecisionAction.ADVANCE,
                subtask_id=subtask.id,
                skill=subtask.skill,
                reason=progress.reason,
                progress_status=progress.status,
                retry_count=retry_count,
                details=progress.metrics,
            )

        requested_skill = progress.metrics.get("requested_skill")
        active_controller_skill = (
            context.world_state
            .get("mission", {})
            .get("active_controller_skill")
        )
        if (
            progress.status == ProgressStatus.RUNNING
            and requested_skill
            and requested_skill != active_controller_skill
        ):
            return OrchestrationDecision(
                action=DecisionAction.SWITCH_SKILL,
                subtask_id=subtask.id,
                skill=subtask.skill,
                reason=progress.reason,
                progress_status=progress.status,
                retry_count=retry_count,
                details=progress.metrics,
            )

        if progress.status == ProgressStatus.RUNNING:
            return OrchestrationDecision(
                action=DecisionAction.CONTINUE_CURRENT,
                subtask_id=subtask.id,
                skill=subtask.skill,
                reason=progress.reason,
                progress_status=progress.status,
                retry_count=retry_count,
                details=progress.metrics,
            )

        if progress.status == ProgressStatus.BLOCKED and retry_count < self.max_blocked_retries:
            return OrchestrationDecision(
                action=DecisionAction.RETRY,
                subtask_id=subtask.id,
                skill=subtask.skill,
                reason=progress.reason,
                progress_status=progress.status,
                retry_count=retry_count + 1,
                details=progress.metrics,
            )

        # Terminal failures and repeatedly blocked states are routed to the
        # future replanning path instead of silently advancing to the next task.
        return OrchestrationDecision(
            action=DecisionAction.REQUEST_REPLAN,
            subtask_id=subtask.id,
            skill=subtask.skill,
            reason=progress.reason,
            progress_status=progress.status,
            retry_count=retry_count,
            details=progress.metrics,
        )
