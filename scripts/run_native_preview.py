"""Preview the native Smart Captain runtime plan and model registry."""

from __future__ import annotations

import json

from smart_captain.app.native_runtime import preview_native_runtime


def main() -> None:
    print(json.dumps(preview_native_runtime(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
