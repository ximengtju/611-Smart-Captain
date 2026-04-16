"""Future unified application entrypoint.

This entrypoint still avoids touching the legacy runtime, but it now exercises
the new planning and orchestration stack end-to-end.
"""

from __future__ import annotations

from smart_captain.app.bridge import LegacyModeBridge


def main() -> None:
    """Run a lightweight planning demo against the new framework skeleton."""
    bridge = LegacyModeBridge()
    plan = bridge.build_plan(
        command="请控制水下机器人先导航到目标区域，途中避障，然后搜索可疑目标，最后切换声呐进行精细测绘",
        mission_id="demo_app_plan",
        world_state={"source": "app.main"},
    )
    print(f"mission_id={plan.mission_id}")
    print(f"skills={plan.skill_sequence}")
    print(f"mode_sequence={plan.mode_sequence}")


if __name__ == "__main__":
    main()
