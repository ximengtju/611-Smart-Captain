"""Preview the new mission runner outputs."""
# 脚本入口层
# $env:PYTHONPATH="src"
# 参考指令，python scripts\run_mission_runner.py --mode execute --command "先导航到目标点，进一步的执行避障任务，躲避障碍物并保持航行安全"

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from smart_captain.app.mission_runner import MissionRequest, MissionRunner


def _default_artifact_paths() -> tuple[Path, Path]:
    """Build timestamped output paths for execute-mode artifacts."""
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"mission_runner_execute_{timestamp}"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.log"

class _TeeWriter:

    """Write runtime progress to both the terminal and a log file.

    `execute` mode prints useful live controller diagnostics from deep inside
    the runtime loop.  We keep those diagnostics visible in PowerShell while
    duplicating the exact same text into the runtime log for later comparison.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        """Forward one print chunk to every configured output stream."""
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        """Flush all streams so long-running execute logs appear promptly."""
        for stream in self.streams:
            stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the new Smart Captain mission runner.")
    # 命令行解析
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
    parser.add_argument(
        "--output",
        default=None,
        help="Write execute-mode JSON summary to this path. Defaults to outputs/<timestamp>.json.",
    )
    parser.add_argument(
        "--runtime-log",
        default=None,
        help="Write execute-mode step/progress logs to this path. Defaults to outputs/<timestamp>.log.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also print the full JSON summary to the terminal after writing files.",
    )
    parser.add_argument(
        "--approach-radius",
        type=float,
        default=None,
        help=(
            "Override the navigation approach-zone radius recorded by the feedback evaluator. "
            "This is telemetry only; navigation still advances on goal_reached."
        ),
    )
    parser.add_argument(
        "--blocked-distance",
        type=float,
        default=None,
        help="Override the obstacle evaluator distance below which the path is treated as blocked.",
    )
    parser.add_argument(
        "--feedback-record-interval",
        type=int,
        default=50,
        help=(
            "Record non-event feedback/continue decisions every N steps. "
            "Evaluation still runs every step."
        ),
    )
    args = parser.parse_args()

    world_state = {
        "source": "run_mission_runner",
        "mission": {},
    }
    if args.approach_radius is not None:
        world_state["mission"]["approach_radius"] = args.approach_radius
    if args.blocked_distance is not None:
        world_state["mission"]["blocked_distance"] = args.blocked_distance
    world_state["mission"]["feedback_record_interval"] = args.feedback_record_interval

    runner = MissionRunner()
    request = MissionRequest(
        mission_id="mission_runner_demo",
        command=args.command,
        world_state=world_state,
    )

    if args.mode == "execute":
        default_json_path, default_log_path = _default_artifact_paths()
        output_path = Path(args.output) if args.output else default_json_path
        runtime_log_path = Path(args.runtime_log) if args.runtime_log else default_log_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Execute mode can print thousands of simulator progress lines.  Capture
        # them in a log file while also teeing them to the terminal, because the
        # live goal/position/speed trace is useful for judging handoff behavior.
        try:
            with runtime_log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"command={args.command}\n")
                log_file.write(f"mode={args.mode}\n\n")
                with redirect_stdout(_TeeWriter(sys.stdout, log_file)):
                    summary = runner.execute(request)
        except Exception:
            print(f"execute failed; runtime log written to: {runtime_log_path}")
            raise

        output_text = json.dumps(summary, ensure_ascii=False, indent=2)
        output_path.write_text(output_text, encoding="utf-8")
        if args.print_json:
            print(output_text)
        else:
            print(f"execute summary written to: {output_path}")
            print(f"runtime log written to: {runtime_log_path}")
        return
    elif args.mode == "legacy-preview":
        summary = runner.legacy_preview(request)
    else:
        summary = runner.plan_summary(request)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
