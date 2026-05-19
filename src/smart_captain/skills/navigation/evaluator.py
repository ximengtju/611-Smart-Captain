"""Progress evaluator for the navigation skill.

Navigation is treated as the global approach phase in the current project.
The approach radius is recorded as feedback, but continuous task chaining only
advances after the navigation environment reports that the goal is reached.
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
class NavigationEvaluator:
    """Evaluate whether the navigation subtask should continue or finish."""

    # Keep the default handoff threshold aligned with the Task1 navigation
    # training success tolerance.  A wider default, such as 10 m, switches to
    # obstacle avoidance before the navigation policy has reached the pose it
    # was previously allowed to settle into, which changes the obstacle model's
    # starting state and can make the demo less stable.
    default_approach_radius: float = 1.3
    default_obstacle_switch_distance: float = 8.0

    def _approach_radius(self, subtask: Subtask, context: ExecutionContext) -> float:
        """Resolve the approach radius from subtask settings or mission state.

        The default keeps the first closed-loop version useful even before LLM
        decomposition starts producing explicit numeric radii.
        """
        if "approach_radius" in subtask.constraints:
            return _read_number(subtask.constraints["approach_radius"], self.default_approach_radius)
        if "approach_radius" in subtask.params:
            return _read_number(subtask.params["approach_radius"], self.default_approach_radius)
        mission_state = context.world_state.get("mission", {})
        return _read_number(mission_state.get("approach_radius"), self.default_approach_radius)

    def _obstacle_switch_distance(self, subtask: Subtask, context: ExecutionContext) -> float:
        """Resolve the near-obstacle controller switch threshold."""
        if "obstacle_switch_distance" in subtask.constraints:
            return _read_number(
                subtask.constraints["obstacle_switch_distance"],
                self.default_obstacle_switch_distance,
            )
        if "obstacle_switch_distance" in subtask.params:
            return _read_number(
                subtask.params["obstacle_switch_distance"],
                self.default_obstacle_switch_distance,
            )
        mission_state = context.world_state.get("mission", {})
        return _read_number(
            mission_state.get("obstacle_switch_distance"),
            self.default_obstacle_switch_distance,
        )

    def evaluate(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        feedback: FeedbackSnapshot,
    ) -> TaskProgress:
        """Translate navigation feedback into dispatcher-ready progress."""
        approach_radius = self._approach_radius(subtask, context)
        obstacle_switch_distance = self._obstacle_switch_distance(subtask, context)
        previous_distance = (
            context.world_state
            .get("skills", {})
            .get(subtask.skill, {})
            .get("distance_to_goal")
        )
        distance_delta = None
        if previous_distance is not None and feedback.distance_to_goal is not None:
            distance_delta = _read_number(previous_distance, feedback.distance_to_goal) - feedback.distance_to_goal
        env_state = context.world_state.get("environment", {})

        # 碰撞恢复
        collision_recovery_active = bool(
            getattr(
                feedback,
                "collision_recovery_active",
                env_state.get("collision_recovery_active", False),
            )
        )

        collision_recovery_failed = bool(
            getattr(
                feedback,
                "collision_recovery_failed",
                env_state.get("collision_recovery_failed", False),
            )
        )

        collision_recovery_step = int(
            getattr(
                feedback,
                "collision_recovery_step",
                env_state.get("collision_recovery_step", 0),
            ) or 0
        )

        collision_count = int(
            getattr(
                feedback,
                "collision_count",
                env_state.get("collision_count", 0),
            ) or 0
        )

        # metrics = {
        #     "distance_to_goal": feedback.distance_to_goal,
        #     "approach_radius": approach_radius,
        #     "distance_delta": distance_delta,
        #     "collision": feedback.collision,
        #     "done": feedback.done,
        #     "truncated": feedback.truncated,
        #     "nearest_obstacle_distance": feedback.nearest_obstacle_distance,
        #     "obstacle_switch_distance": obstacle_switch_distance,
        # }
        metrics = {
            "distance_to_goal": feedback.distance_to_goal,
            "approach_radius": approach_radius,
            "distance_delta": distance_delta,
            "collision": feedback.collision,
            "done": feedback.done,
            "truncated": feedback.truncated,
            "nearest_obstacle_distance": feedback.nearest_obstacle_distance,
            "obstacle_switch_distance": obstacle_switch_distance,
            "collision_recovery_active": collision_recovery_active,
            "collision_recovery_failed": collision_recovery_failed,
            "collision_recovery_step": collision_recovery_step,
            "collision_count": collision_count,
        }

        # skill_state = {
        #     "navigation_phase": "approach",
        #     "distance_to_goal": feedback.distance_to_goal,
        #     "approach_radius": approach_radius,
        #     "distance_delta": distance_delta,
        # }

        skill_state = {
            "navigation_phase": "approach",
            "distance_to_goal": feedback.distance_to_goal,
            "approach_radius": approach_radius,
            "distance_delta": distance_delta,
            "collision_recovery_active": collision_recovery_active,
            "collision_recovery_failed": collision_recovery_failed,
            "collision_recovery_step": collision_recovery_step,
            "collision_count": collision_count,
        }

        in_approach_zone = (
            feedback.distance_to_goal is not None
            and feedback.distance_to_goal <= approach_radius
        )
        near_obstacle = (
            feedback.nearest_obstacle_distance is not None
            and feedback.nearest_obstacle_distance < obstacle_switch_distance
        )
        skill_state["in_approach_zone"] = in_approach_zone
        skill_state["near_obstacle"] = near_obstacle

        # if feedback.collision:
        #     status = ProgressStatus.FAILED
        #     reason = "navigation_collision"
        # elif feedback.goal_reached:
        #     status = ProgressStatus.SUCCEEDED
        #     reason = "navigation_goal_reached"
        # elif feedback.truncated:
        #     status = ProgressStatus.FAILED
        #     reason = "navigation_timeout"
        # elif feedback.done:
        #     status = ProgressStatus.FAILED
        #     reason = "navigation_terminal_without_goal"
        # elif near_obstacle:
        #     status = ProgressStatus.RUNNING
        #     reason = "navigation_near_obstacle_switch_to_avoidance"
        #     metrics.update({
        #         "requested_skill": "obstacle_avoidance",
        #         "preserve_goal": True,
        #     })
        #     skill_state.update({
        #         "requested_skill": "obstacle_avoidance",
        #         "preserve_goal": True,
        #     })
        # elif in_approach_zone:
        #     # Entering the approach radius is useful feedback, but this project
        #     # uses continuous model chaining: the next skill must inherit the
        #     # navigation model's terminal state, not an early intermediate
        #     # state.  Therefore this remains RUNNING until the env reports
        #     # `goal_reached`.
        #     status = ProgressStatus.RUNNING
        #     reason = "navigation_in_approach_zone_waiting_for_goal"
        # else:
        #     status = ProgressStatus.RUNNING
        #     reason = "navigation_in_progress"

        # 先进行碰撞恢复的判定
        if collision_recovery_failed:
            status = ProgressStatus.FAILED
            reason = "navigation_collision_recovery_failed"

        elif collision_recovery_active:
             status = ProgressStatus.RUNNING
             reason = "navigation_collision_recovery_active"

        elif feedback.collision:
            status = ProgressStatus.FAILED
            reason = "navigation_collision"

        elif feedback.goal_reached:
            status = ProgressStatus.SUCCEEDED
            reason = "navigation_goal_reached"

        elif feedback.truncated:
            status = ProgressStatus.FAILED
            reason = "navigation_timeout"

        elif feedback.done:
            status = ProgressStatus.FAILED
            reason = "navigation_terminal_without_goal"

        elif near_obstacle:
            status = ProgressStatus.RUNNING
            reason = "navigation_near_obstacle_switch_to_avoidance"
            metrics.update({
                "requested_skill": "obstacle_avoidance",
                "preserve_goal": True,
            })
            skill_state.update({
                "requested_skill": "obstacle_avoidance",
                "preserve_goal": True,
            })

        elif in_approach_zone:
            status = ProgressStatus.RUNNING
            reason = "navigation_in_approach_zone_waiting_for_goal"

        else:
            status = ProgressStatus.RUNNING
            reason = "navigation_in_progress"


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
