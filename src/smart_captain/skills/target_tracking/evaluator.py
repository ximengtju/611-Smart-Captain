from __future__ import annotations

from smart_captain.orchestration.feedback import (
    FeedbackSnapshot,
    ProgressStatus,
    TaskProgress,
)


class TargetTrackingEvaluator:
    def evaluate(self, *, subtask, context, feedback: FeedbackSnapshot) -> TaskProgress:
        if feedback.collision or feedback.truncated:
            status = ProgressStatus.FAILED
            reason = "target_tracking_terminal_failure"
        elif feedback.goal_reached or feedback.done:
            status = ProgressStatus.SUCCEEDED
            reason = "target_tracking_done"
        else:
            status = ProgressStatus.RUNNING
            reason = "target_tracking_running"

        return TaskProgress(
            subtask_id=subtask.id,
            skill=subtask.skill,
            status=status,
            reason=reason,
            confidence=0.6,
            metrics={
                "done": feedback.done,
                "truncated": feedback.truncated,
                "collision": feedback.collision,
                "goal_reached": feedback.goal_reached,
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