"""Preview the new mission runner outputs."""

from __future__ import annotations

import argparse
import json

from smart_captain.app.mission_runner import MissionRequest, MissionRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the new Smart Captain mission runner.")
    parser.add_argument(
        "--mode",
        choices=("plan-only", "legacy-preview", "execute"),
        default="plan-only",
        help="Which mission-runner view to output.",
    )
    parser.add_argument(
        "--command",
        default="请控制水下机器人先导航到目标区域，途中避障，然后搜索可疑目标，最后切换声呐进行精细测绘",
        help="Mission command to parse.",
    )
    args = parser.parse_args()

    runner = MissionRunner()
    request = MissionRequest(
        mission_id="mission_runner_demo",
        command=args.command,
        world_state={"source": "run_mission_runner"},
    )

    if args.mode == "execute":
        summary = runner.execute(request)
    elif args.mode == "legacy-preview":
        summary = runner.legacy_preview(request)
    else:
        summary = runner.plan_summary(request)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
