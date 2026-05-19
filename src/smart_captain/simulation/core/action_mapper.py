"""Shared action-to-thruster mapping utilities.

These helpers are extracted from the current project logic but are not yet
wired into the legacy runtime. They define a stable home for future migration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CANONICAL_ACTION_LOW = np.array([-28.5, -20.0, -20.0, -10.0], dtype=np.float32)
CANONICAL_ACTION_HIGH = np.array([28.5, 20.0, 20.0, 10.0], dtype=np.float32)


@dataclass(frozen=True)
class HoveringActionMapper:
    """Map canonical low-level actions to the 8-thruster HoloOcean command."""

    thruster_count: int = 8
    command_limit: float = 28.5

    def to_command(self, action: np.ndarray) -> np.ndarray:
        """Convert canonical `[surge, sway, heave, yaw]` actions to thruster commands."""
        surge, sway, heave, yaw = np.asarray(action, dtype=np.float32)

        command = np.zeros(self.thruster_count, dtype=np.float32)

        command[0:4] = heave
        command[4] = surge - sway + yaw
        command[5] = surge + sway - yaw
        command[6] = surge - sway - yaw
        command[7] = surge + sway + yaw
        command = np.clip(command, -self.command_limit, self.command_limit)
        return command


DEFAULT_ACTION_MAPPER = HoveringActionMapper()
