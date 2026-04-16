"""Runtime descriptors and adapters for the legacy shared-AUV multi-task pattern."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LegacyTaskBinding:
    """Describe how one legacy mode maps to a task environment."""

    mode_index: int
    skill_name: str
    env_import_path: str
    shares_runtime_auv: bool = True


@dataclass
class LegacySharedAUVLayout:
    """Describe the task-combine1 shared-AUV execution pattern.

    The old runtime builds multiple task environments on top of one HoloOcean
    instance and synchronizes state when switching modes. This object gives that
    pattern a stable representation inside the new framework.
    """

    scenario_cfg: dict[str, Any]
    task_bindings: list[LegacyTaskBinding] = field(default_factory=list)
    sync_state_on_switch: bool = True

    def mode_to_skill(self) -> dict[int, str]:
        """Return the legacy mode-to-skill mapping."""
        return {binding.mode_index: binding.skill_name for binding in self.task_bindings}

    def skill_to_mode(self) -> dict[str, int]:
        """Return the inverse mapping for registered legacy tasks."""
        return {binding.skill_name: binding.mode_index for binding in self.task_bindings}


def load_symbol(import_path: str):
    """Load a symbol from a `module:symbol` import path."""
    module_name, symbol_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


@dataclass
class LegacySharedAUVRuntimeAdapter:
    """New-framework adapter for the old shared-AUV multi-env execution model.

    This mirrors the semantics of `Main-Framework/task/task_combine1.py`, but
    creates task environments from a declarative binding layout instead of
    hard-coding them in one legacy file.
    """

    shared_auv: Any
    layout: LegacySharedAUVLayout
    env_config: dict[str, Any]
    mode: int = 0
    task_env: list[Any] = field(default_factory=list)

    def bootstrap(self) -> "LegacySharedAUVRuntimeAdapter":
        """Instantiate all bound task environments on top of one shared AUV."""
        ordered_bindings = sorted(self.layout.task_bindings, key=lambda binding: binding.mode_index)
        self.task_env = []
        for binding in ordered_bindings:
            env_cls = load_symbol(binding.env_import_path)
            runtime_env = env_cls(self.env_config, self.shared_auv, False)
            self.task_env.append(runtime_env)
        return self

    def reset(self, seed=None, return_info: bool = True, options=None):
        """Reset the currently active task environment."""
        return self.task_env[self.mode].reset(seed, return_info, options)

    def step(self, action):
        """Step the currently active task environment."""
        return self.task_env[self.mode].step(action)

    def set_multi_mode_index(self, index: int) -> None:
        """Switch mode and synchronize state if the target env supports it."""
        if index < 0 or index >= len(self.task_env):
            raise IndexError(f"Invalid mode index {index}, available modes: {len(self.task_env)}")
        last_mode = self.mode
        self.mode = index
        if self.layout.sync_state_on_switch and last_mode != self.mode:
            last_env = self.task_env[last_mode]
            next_env = self.task_env[self.mode]
            if hasattr(next_env, "sync_state_from"):
                next_env.sync_state_from(last_env)
