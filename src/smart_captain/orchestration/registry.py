"""Registry helpers used by the orchestration layer."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

from smart_captain.simulation.registry import SCENARIO_REGISTRY, SENSOR_REGISTRY
from smart_captain.skills.registry import SKILL_REGISTRY


def _split_import_path(path: str) -> tuple[str, str]:
    module_name, symbol_name = path.split(":", 1)
    return module_name, symbol_name


@lru_cache(maxsize=128)
def load_symbol(import_path: str) -> Any:
    """Load a class or symbol from a `module:symbol` string."""
    module_name, symbol_name = _split_import_path(import_path)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def resolve_skill_spec(skill_name: str):
    """Resolve a skill specification by name."""
    if skill_name not in SKILL_REGISTRY:
        raise KeyError(f"Unknown skill: {skill_name}")
    return SKILL_REGISTRY[skill_name]


def resolve_scenario_class(scenario_name: str):
    """Resolve a scenario class by registry key."""
    if scenario_name not in SCENARIO_REGISTRY:
        raise KeyError(f"Unknown scenario: {scenario_name}")
    return load_symbol(SCENARIO_REGISTRY[scenario_name])


def resolve_sensor_class(sensor_name: str):
    """Resolve a sensor adapter class by registry key."""
    if sensor_name not in SENSOR_REGISTRY:
        raise KeyError(f"Unknown sensor: {sensor_name}")
    return load_symbol(SENSOR_REGISTRY[sensor_name])


def resolve_skill_env_class(skill_name: str):
    """Resolve the environment adapter class for a skill."""
    spec = resolve_skill_spec(skill_name)
    return load_symbol(spec.env_cls)


def resolve_skill_policy_class(skill_name: str):
    """Resolve the policy wrapper class for a skill."""
    spec = resolve_skill_spec(skill_name)
    return load_symbol(spec.policy_cls)
