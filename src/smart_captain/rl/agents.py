"""RL execution layer for multi-skill policy switching."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from smart_captain.rl.model_store import ModelSpec, ModelStore


class PredictiveModel(Protocol):
    """Minimal protocol needed from a loaded RL model."""

    def predict(self, obs, state=None, deterministic: bool = True):
        """Return `(action, next_state)`."""


class ModelLoader(Protocol):
    """Protocol for algorithm-specific model loading."""

    def load(self, path: str):
        """Load a model from disk."""


@dataclass
class SkillPolicyRuntime:
    """Loaded policy runtime for one skill."""

    skill: str
    spec: ModelSpec
    model: PredictiveModel


@dataclass
class MultiSkillAgentRuntime:
    """Multi-policy runtime with explicit skill switching."""

    model_store: ModelStore
    algorithm_loaders: dict[str, ModelLoader]
    runtimes: dict[str, SkillPolicyRuntime] = field(default_factory=dict)
    active_skill: str | None = None
    skill_to_mode: dict[str, int] = field(default_factory=dict)
    mode_to_skill: dict[int, str] = field(default_factory=dict)

    def ensure_loaded(self, skill: str) -> SkillPolicyRuntime:
        """Load and cache the policy runtime for a skill."""
        if skill in self.runtimes:
            return self.runtimes[skill]

        spec = self.model_store.get(skill)
        if spec.algorithm not in self.algorithm_loaders:
            raise KeyError(f"No loader registered for algorithm '{spec.algorithm}'")
        loader = self.algorithm_loaders[spec.algorithm]
        model = loader.load(str(spec.absolute_path))
        runtime = SkillPolicyRuntime(skill=skill, spec=spec, model=model)
        self.runtimes[skill] = runtime
        return runtime

    def activate(self, skill: str) -> SkillPolicyRuntime:
        """Activate one skill policy."""
        runtime = self.ensure_loaded(skill)
        self.active_skill = skill
        return runtime

    def register_mode_mapping(self, skill: str, mode_index: int) -> None:
        """Register a legacy-style integer mode for one skill."""
        self.skill_to_mode[skill] = mode_index
        self.mode_to_skill[mode_index] = skill

    def activate_mode(self, mode_index: int) -> SkillPolicyRuntime:
        """Activate the skill registered for a legacy-style mode index."""
        if mode_index not in self.mode_to_skill:
            raise KeyError(f"No skill registered for mode index '{mode_index}'")
        return self.activate(self.mode_to_skill[mode_index])

    def predict(self, obs, state=None, deterministic: bool = True):
        """Predict an action with the currently active skill runtime."""
        if self.active_skill is None:
            raise RuntimeError("No active skill. Call activate(skill) before predict().")
        runtime = self.ensure_loaded(self.active_skill)
        return runtime.model.predict(obs, state=state, deterministic=deterministic)

    def has_skill(self, skill: str) -> bool:
        """Return whether the model store knows this skill."""
        return self.model_store.has(skill)

    def has_mode(self, mode_index: int) -> bool:
        """Return whether the runtime knows this legacy mode index."""
        return mode_index in self.mode_to_skill


@dataclass
class LegacyCompatibleAgents:
    """Compatibility layer that mirrors the old `Main-Framework/env/agents.py`.

    This is the migration target for callers that still think in terms of
    `model_paths`, `model_types`, `mode='single'|'multi'`, and integer mode
    switching.
    """

    model_paths: str | list[str]
    model_types: str | list[str]
    model_class_dict: dict[str, Any]
    mode: str = "single"
    skill_names: list[str] | None = None
    models: list[PredictiveModel] = field(default_factory=list)
    mode_to_model_map: dict[str, int] = field(default_factory=dict)
    current_multi_mode_index: int = 0

    def __post_init__(self) -> None:
        paths = [self.model_paths] if isinstance(self.model_paths, str) else list(self.model_paths)
        types = [self.model_types] if isinstance(self.model_types, str) else list(self.model_types)
        if len(paths) != len(types):
            raise ValueError("model_paths and model_types must have the same length")

        for path, model_type in zip(paths, types):
            if model_type not in self.model_class_dict:
                raise ValueError(f"Unsupported model type: {model_type}")
            model_class = self.model_class_dict[model_type]
            self.models.append(model_class.load(path))

        if self.mode == "single" and len(self.models) != 1:
            self.current_multi_mode_index = 0

        if self.skill_names is not None and len(self.skill_names) != len(self.models):
            raise ValueError("skill_names must align one-to-one with loaded models")

    def set_mode(self, mode: str) -> None:
        """Switch between single-model and multi-model mode."""
        self.mode = mode

    def set_multi_mode_index(self, index: int) -> None:
        """Activate a model by integer index."""
        if index < 0 or index >= len(self.models):
            raise IndexError(f"Invalid model index {index}, available models: {len(self.models)}")
        self.current_multi_mode_index = index

    def set_multi_mode_by_key(self, key: str) -> None:
        """Activate a model using a string mapping key."""
        if key not in self.mode_to_model_map:
            raise KeyError(f"Unknown mode mapping key: {key}")
        self.current_multi_mode_index = self.mode_to_model_map[key]

    def add_mode_mapping(self, key: str, model_index: int) -> None:
        """Register a string key to model-index mapping."""
        if model_index < 0 or model_index >= len(self.models):
            raise IndexError(f"Invalid model index {model_index}, available models: {len(self.models)}")
        self.mode_to_model_map[key] = model_index

    def predict(self, obs, state: Any | None = None, deterministic: bool = True):
        """Predict with the active model using the old selection semantics."""
        if self.mode == "single":
            model = self.models[0]
        else:
            model = self.models[self.current_multi_mode_index]
        return model.predict(obs, state=state, deterministic=deterministic)

    @property
    def active_skill(self) -> str | None:
        """Return the current skill name when provided."""
        if self.skill_names is None:
            return None
        return self.skill_names[self.current_multi_mode_index]


def build_legacy_compatible_agents_from_store(
    model_store: ModelStore,
    model_class_dict: dict[str, Any],
    skills: list[str],
    mode: str = "multi",
) -> LegacyCompatibleAgents:
    """Build a compatibility agent wrapper from registered skill models."""
    model_paths: list[str] = []
    model_types: list[str] = []
    for skill in skills:
        spec = model_store.get(skill)
        model_paths.append(str(spec.absolute_path))
        model_types.append(spec.algorithm)

    return LegacyCompatibleAgents(
        model_paths=model_paths,
        model_types=model_types,
        model_class_dict=model_class_dict,
        mode=mode,
        skill_names=skills,
    )
