"""Shared action-to-thruster mapping utilities.

These helpers are extracted from the current project logic but are not yet
wired into the legacy runtime. They define a stable home for future migration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HoveringActionMapper:
    """Map canonical low-level actions to the 8-thruster HoloOcean command."""

    thruster_count: int = 8

    def to_command(self, action: np.ndarray) -> np.ndarray:
        """Convert `[surge, sway, yaw, heave]` to thruster commands.

        This preserves the mapping currently used in
        `Main-Framework/env/pierharbor_hovering.py`.
        """
        surge, sway, yaw, heave = np.asarray(action, dtype=np.float32)

        command = np.zeros(self.thruster_count, dtype=np.float32)

        command[4:8] += surge
        command[4] += sway
        command[5] -= sway
        command[6] += sway
        command[7] -= sway

        command[4] += yaw
        command[5] -= yaw
        command[6] -= yaw
        command[7] += yaw

        command[0:4] = heave
        return command


DEFAULT_ACTION_MAPPER = HoveringActionMapper()
