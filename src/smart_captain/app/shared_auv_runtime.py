"""Shared-AUV runtime entrypoints implemented inside the new framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_captain.orchestration.multi_env import (
    LegacySharedAUVLayout,
    LegacySharedAUVRuntimeAdapter,
)
from smart_captain.simulation.defaults import DEFAULT_ENV_CONFIG


@dataclass
class SharedAUVMissionRuntime:
    """Compatibility runtime that mirrors the old multi-task shared-AUV loop."""

    adapter: LegacySharedAUVRuntimeAdapter

    def reset(self, seed=None, return_info: bool = True, options=None):
        """Reset the currently active task environment."""
        return self.adapter.reset(seed, return_info, options)

    def step(self, action):
        """Step the currently active task environment."""
        return self.adapter.step(action)

    def set_multi_mode_index(self, index: int) -> None:
        """Switch active task mode and synchronize state."""
        self.adapter.set_multi_mode_index(index)

    @property
    def mode(self) -> int:
        """Expose the active legacy mode index."""
        return self.adapter.mode


def create_shared_auv_runtime(
    layout: LegacySharedAUVLayout,
    env_config: dict[str, Any] = DEFAULT_ENV_CONFIG,
    mode: int = 0,
    show_viewport: bool = True,
) -> SharedAUVMissionRuntime:
    """Create a shared HoloOcean runtime from the new declarative layout."""
    import holoocean

    scenario_cfg = env_config["auv_config"]
    shared_auv = holoocean.make(scenario_cfg=scenario_cfg, show_viewport=show_viewport)
    adapter = LegacySharedAUVRuntimeAdapter(
        shared_auv=shared_auv,
        layout=layout,
        env_config=env_config,
        mode=mode,
    ).bootstrap()
    return SharedAUVMissionRuntime(adapter=adapter)
