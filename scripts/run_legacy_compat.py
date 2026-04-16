"""Preview or bootstrap the legacy runtime using the new planning layer."""

from __future__ import annotations

import argparse
import json

from smart_captain.app.legacy_runtime import execute_legacy_plan, preview_legacy_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or preview the legacy compatibility runtime.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually initialize the legacy runtime and execute the mission loop.",
    )
    args = parser.parse_args()

    summary = execute_legacy_plan() if args.execute else preview_legacy_plan()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
