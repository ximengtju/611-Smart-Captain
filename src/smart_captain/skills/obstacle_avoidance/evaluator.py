"""Progress evaluator for the obstacle-avoidance skill.

Obstacle avoidance is treated as the local safety phase after navigation has
reached the target area.  The evaluator checks both goal completion and safety
signals so the dispatcher can distinguish success, collision failure, and a
temporarily blocked path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_captain.orchestration.feedback import FeedbackSnapshot, ProgressStatus, TaskProgress
from smart_captain.orchestration.task_graph import ExecutionContext, Subtask


def _read_number(value: Any, default: float) -> float:
    """Read numeric configuration values defensively from task/context dicts."""
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class ObstacleAvoidanceEvaluator:
    """Evaluate local goal completion and obstacle-related safety."""

    default_goal_radius: float = 1.5
    default_safe_distance: float = 3.0
    default_blocked_distance: float = 0.0

    def _threshold(self, subtask: Subtask, context: ExecutionContext, name: str, default: float) -> float:
        """Resolve evaluator thresholds from subtask constraints or mission state."""
        if name in subtask.constraints:
            return _read_number(subtask.constraints[name], default)
        if name in subtask.params:
            return _read_number(subtask.params[name], default)
        mission_state = context.world_state.get("mission", {})
        return _read_number(mission_state.get(name), default)

    def evaluate(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        feedback: FeedbackSnapshot,
    ) -> TaskProgress:
        """Translate obstacle-avoidance feedback into task progress."""
        goal_radius = self._threshold(subtask, context, "goal_radius", self.default_goal_radius)
        safe_distance = self._threshold(subtask, context, "safe_distance", self.default_safe_distance)
        blocked_distance = self._threshold(
            subtask,
            context,
            "blocked_distance",
            self.default_blocked_distance,
        )

        nearest = feedback.nearest_obstacle_distance
        safe_clearance = nearest is None or nearest >= safe_distance
        blocked_clearance = nearest is not None and nearest < blocked_distance
        reached_goal = feedback.goal_reached or (
            feedback.distance_to_goal is not None
            and feedback.distance_to_goal <= goal_radius
        )

        metrics = {
            "distance_to_goal": feedback.distance_to_goal,
            "goal_radius": goal_radius,
            "nearest_obstacle_distance": nearest,
            "safe_distance": safe_distance,
            "blocked_distance": blocked_distance,
            "collision": feedback.collision,
            "done": feedback.done,
            "truncated": feedback.truncated,
        }

        skill_state = {
            "avoidance_phase": "local_approach",
            "distance_to_goal": feedback.distance_to_goal,
            "goal_radius": goal_radius,
            "nearest_obstacle_distance": nearest,
            "safe_distance": safe_distance,
            "blocked_distance": blocked_distance,
        }

        if feedback.collision:
            status = ProgressStatus.FAILED
            reason = "obstacle_avoidance_collision"
        elif reached_goal and safe_clearance:
            status = ProgressStatus.SUCCEEDED
            reason = "obstacle_avoidance_safe_goal_reached"
        elif blocked_clearance:
            status = ProgressStatus.BLOCKED
            reason = "obstacle_avoidance_blocked_clearance"
        elif feedback.truncated:
            status = ProgressStatus.FAILED
            reason = "obstacle_avoidance_timeout"
        elif feedback.done and not reached_goal:
            status = ProgressStatus.FAILED
            reason = "obstacle_avoidance_terminal_without_goal"
        else:
            status = ProgressStatus.RUNNING
            reason = "obstacle_avoidance_in_progress"

        skill_state.update({"status": status.value, "status_reason": reason})
        return TaskProgress(
            subtask_id=subtask.id,
            skill=subtask.skill,
            status=status,
            reason=reason,
            confidence=1.0 if feedback.distance_to_goal is not None else 0.5,
            metrics=metrics,
            world_state_delta={"skills": {subtask.skill: skill_state}},
        )
