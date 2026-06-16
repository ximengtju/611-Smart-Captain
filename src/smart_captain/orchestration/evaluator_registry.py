"""Registry for skill progress evaluators.

The orchestration layer resolves evaluators by skill name in the same spirit
as it resolves env and policy classes.  This keeps the dispatcher generic:
adding a new skill later should require a new evaluator, not a dispatcher
rewrite.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from smart_captain.orchestration.feedback import FeedbackSnapshot, ProgressStatus, TaskProgress
from smart_captain.orchestration.registry import load_symbol
from smart_captain.orchestration.task_graph import ExecutionContext, Subtask


EVALUATOR_REGISTRY: dict[str, str] = {
    "navigation": "smart_captain.skills.navigation.evaluator:NavigationEvaluator",
    "obstacle_avoidance": (
        "smart_captain.skills.obstacle_avoidance.evaluator:"
        "ObstacleAvoidanceEvaluator"
    ),
    "target_tracking": "smart_captain.skills.target_tracking.evaluator:TargetTrackingEvaluator",
}


@lru_cache(maxsize=32)
def get_skill_evaluator(skill_name: str) -> Any:
    """Instantiate the evaluator registered for `skill_name`."""
    if skill_name not in EVALUATOR_REGISTRY:
        raise KeyError(f"No progress evaluator registered for skill '{skill_name}'")
    evaluator_cls = load_symbol(EVALUATOR_REGISTRY[skill_name])
    return evaluator_cls()


def evaluate_skill_progress(
    *,
    subtask: Subtask,
    context: ExecutionContext,
    feedback: FeedbackSnapshot,
) -> TaskProgress:
    """Evaluate the active subtask with its registered skill evaluator."""
    try:
        evaluator = get_skill_evaluator(subtask.skill)
    except KeyError:
        # During migration only navigation and obstacle avoidance have
        # skill-specific evaluators.  The generic fallback preserves old
        # done-driven behavior for placeholder skills until they gain their own
        # feedback semantics.
        if feedback.goal_reached:
            status = ProgressStatus.SUCCEEDED
            reason = "generic_goal_reached"
        elif feedback.collision or feedback.truncated:
            status = ProgressStatus.FAILED
            reason = "generic_terminal_failure"
        elif feedback.done:
            status = ProgressStatus.SUCCEEDED
            reason = "generic_done"
        else:
            status = ProgressStatus.RUNNING
            reason = "generic_running"
        return TaskProgress(
            subtask_id=subtask.id,
            skill=subtask.skill,
            status=status,
            reason=reason,
            confidence=0.5,
            metrics={
                "done": feedback.done,
                "truncated": feedback.truncated,
                "goal_reached": feedback.goal_reached,
                "collision": feedback.collision,
            },
            world_state_delta={
                "skills": {
                    subtask.skill: {
                        "status": status.value,
                        "status_reason": reason,
                    }
                }
            },
        )
    return evaluator.evaluate(subtask=subtask, context=context, feedback=feedback)
