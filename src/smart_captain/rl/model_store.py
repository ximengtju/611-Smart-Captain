"""Model registration and path resolution for RL skills."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ModelSpec:
    """Describe one RL policy artifact."""

    skill: str
    algorithm: str
    relative_path: str
    source: str = "repo"

    @property
    def absolute_path(self) -> Path:
        """Resolve the model path against the repository layout."""
        return REPO_ROOT / self.relative_path


@dataclass
class ModelStore:
    """Registry-backed lookup for RL model artifacts."""

    model_specs: dict[str, ModelSpec] = field(default_factory=lambda: {
        "navigation": ModelSpec(
            skill="navigation",
            algorithm="sac",
            relative_path="models/rl/navigation/sac/task1-v0_SAC_3/task1-v0_SAC_400000.zip",
        ),
        "obstacle_avoidance": ModelSpec(
            skill="obstacle_avoidance",
            algorithm="sac",
            relative_path="models/rl/obstacle_avoidance/sac/task2-v0_SAC_1/task2-v0_SAC_200000.zip",
        ),
    })

    def get(self, skill: str) -> ModelSpec:
        """Return the registered model spec for a skill."""
        if skill not in self.model_specs:
            raise KeyError(f"No model registered for skill '{skill}'")
        return self.model_specs[skill]

    def has(self, skill: str) -> bool:
        """Return whether a model is registered for the skill."""
        return skill in self.model_specs

    def to_summary(self) -> dict[str, dict[str, str]]:
        """Return a serializable view of the current registry."""
        return {
            skill: {
                "algorithm": spec.algorithm,
                "relative_path": spec.relative_path,
                "absolute_path": str(spec.absolute_path),
                "source": spec.source,
            }
            for skill, spec in self.model_specs.items()
        }

    @staticmethod
    def find_latest_model(env_name: str, model_type: str, log_dir: str | Path) -> str | None:
        """Find the latest completed model artifact in a legacy-style log tree."""
        log_dir_path = Path(log_dir)
        if not log_dir_path.exists():
            return None

        matching_dirs = [
            directory for directory in log_dir_path.iterdir()
            if directory.is_dir() and directory.name.startswith(f"{env_name}_{model_type}_")
        ]
        if not matching_dirs:
            return None

        latest_dir = sorted(
            matching_dirs,
            key=lambda directory: int(directory.name.split("_")[-1]),
        )[-1]
        model_files = list(latest_dir.glob("*.zip"))
        if not model_files:
            return None

        try:
            latest_model = sorted(
                model_files,
                key=lambda model_file: int("".join(filter(str.isdigit, model_file.stem))),
            )[-1]
        except Exception:
            latest_model = sorted(model_files, key=os.path.getmtime)[-1]
        return str(latest_model)
