"""Feedback and progress contracts for closed-loop orchestration.

This module intentionally contains only small data objects.  The runtime loop
uses `FeedbackSnapshot` to describe what the simulator just reported, skill
evaluators use `TaskProgress` to describe how the active skill is doing, and
the dispatcher consumes those progress objects without needing to understand
HoloOcean-specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProgressStatus(str, Enum):
    """Normalized task-progress states understood by the dispatcher.

    `RUNNING` means the current skill should keep control.
    `SUCCEEDED` means the current subtask can be marked complete.
    `FAILED` means local execution has hit a terminal problem.
    `BLOCKED` means the skill is not terminal yet, but recovery logic should
    decide whether to keep trying, retry, or ask for a replan.
    """

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class FeedbackSnapshot:
    """One normalized view of the latest environment feedback.

    The raw simulator still returns `obs, reward, done, truncated, info`, but
    the orchestration layer should not need to know where goal distance or
    collision flags are stored on a particular environment class.  This object
    is the stable handoff point from runtime code into evaluator code.
    """

    mission_id: str
    subtask_id: str
    skill: str
    step_count: int
    task_skill: str | None = None
    controller_skill: str | None = None
    observation: Any = None
    reward: float | None = None
    done: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    position: list[float] | None = None
    velocity: list[float] | None = None
    goal: list[float] | None = None
    distance_to_goal: float | None = None
    collision: bool | None = None
    goal_reached: bool | None = None
    nearest_obstacle_distance: float | None = None
    # 碰撞恢复
    collision_recovery_active: bool = False
    collision_recovery_failed: bool = False
    collision_recovery_step: int = 0
    collision_count: int = 0

    def to_world_state_delta(self) -> dict[str, Any]:
        """Return the shared-state update represented by this snapshot.

        The update is deliberately namespaced.  Mission, vehicle, and
        environment fields are shared facts; the `skills.<skill>` branch keeps
        the active skill's latest measurements from overwriting another skill's
        own progress records.
        """
        vehicle: dict[str, Any] = {}
        if self.position is not None:
            vehicle["position"] = self.position
        if self.velocity is not None:
            vehicle["velocity"] = self.velocity

        task_skill = self.task_skill or self.skill
        controller_skill = self.controller_skill or task_skill
        mission: dict[str, Any] = {
            "active_subtask_id": self.subtask_id,
            "active_skill": self.skill,
            "task_skill": task_skill,
            "controller_skill": controller_skill,
            "step_count": self.step_count,
        }
        if self.goal is not None:
            mission["current_goal"] = self.goal

        environment: dict[str, Any] = {}
        if self.collision is not None:
            environment["collision"] = self.collision
        if self.nearest_obstacle_distance is not None:
            environment["nearest_obstacle_distance"] = self.nearest_obstacle_distance

        skill_state: dict[str, Any] = {
            "last_reward": self.reward,
            "done": self.done,
            "truncated": self.truncated,
        }
        if self.distance_to_goal is not None:
            skill_state["distance_to_goal"] = self.distance_to_goal
        if self.goal_reached is not None:
            skill_state["goal_reached"] = self.goal_reached

        return {
            "mission": mission,
            "vehicle": vehicle,
            "environment": environment,
            "skills": {self.skill: skill_state},
        }

    def to_history_record(self) -> dict[str, Any]:
        """Return a compact record suitable for `ExecutionContext` history.

        Observations can be large arrays, so the dispatcher history stores only
        the scalar fields that are useful for debugging decisions and future
        replanning requests.
        """
        return {
            "subtask_id": self.subtask_id,
            "skill": self.skill,
            "task_skill": self.task_skill or self.skill,
            "controller_skill": self.controller_skill or self.task_skill or self.skill,
            "step_count": self.step_count,
            "reward": self.reward,
            "done": self.done,
            "truncated": self.truncated,
            "distance_to_goal": self.distance_to_goal,
            "collision": self.collision,
            "goal_reached": self.goal_reached,
            "nearest_obstacle_distance": self.nearest_obstacle_distance,
        }


@dataclass(frozen=True)
class TaskProgress:
    """Evaluator output consumed by the dispatcher.

    Evaluators translate environment facts into orchestration facts.  They do
    not switch modes directly; they report status, metrics, and state deltas so
    the dispatcher and recovery policy can choose the next action.
    """

    subtask_id: str
    skill: str
    status: ProgressStatus
    reason: str
    confidence: float = 1.0
    metrics: dict[str, Any] = field(default_factory=dict)
    world_state_delta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return a serializable record for context and runtime summaries."""
        return {
            "subtask_id": self.subtask_id,
            "skill": self.skill,
            "status": self.status.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "metrics": self.metrics,
        }
