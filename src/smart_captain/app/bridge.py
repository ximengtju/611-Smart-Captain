"""Compatibility bridge between the new planner and the legacy mode loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smart_captain.llm.interface import StructuredLLMInterface
from smart_captain.orchestration.multi_env import LegacySharedAUVLayout, LegacyTaskBinding
from smart_captain.orchestration.task_graph import TaskGraph


@dataclass(frozen=True)
class LegacyExecutionPlan:
    """Serializable plan that mirrors the old mode-switching runtime."""

    mission_id: str
    command: str
    mode_sequence: list[int]
    skill_sequence: list[str]
    subtask_ids: list[str]
    graph: TaskGraph
    runtime_layout: LegacySharedAUVLayout | None = None


@dataclass
class LegacyModeBridge:
    """Convert structured task graphs into the old integer mode sequence."""

    llm_interface: StructuredLLMInterface = field(default_factory=StructuredLLMInterface)
    skill_to_mode: dict[str, int] = field(default_factory=lambda: {
        "navigation": 0,
        "obstacle_avoidance": 1,
        "target_tracking": 1,
        "search": 0,
        "mapping": 0,
    })
    runtime_layout: LegacySharedAUVLayout = field(default_factory=lambda: LegacySharedAUVLayout(
        scenario_cfg={"source": "smart_captain.simulation.defaults:DEFAULT_ENV_CONFIG['auv_config']"},
        task_bindings=[
            LegacyTaskBinding(
                mode_index=0,
                skill_name="navigation",
                env_import_path="smart_captain.skills.navigation.env:NavigationEnv",
            ),
            LegacyTaskBinding(
                mode_index=1,
                skill_name="obstacle_avoidance",
                env_import_path="smart_captain.skills.obstacle_avoidance.env:ObstacleAvoidanceEnv",
            ),
        ],
        sync_state_on_switch=True,
    ))

    def graph_to_mode_sequence(self, graph: TaskGraph) -> list[int]:
        """Map each subtask in a graph to a legacy mode index."""
        sequence: list[int] = []
        for subtask in graph.subtasks:
            if subtask.skill not in self.skill_to_mode:
                raise KeyError(f"No legacy mode mapping defined for skill '{subtask.skill}'")
            sequence.append(self.skill_to_mode[subtask.skill])
        return sequence

    def build_plan(
        self,
        command: str,
        mission_id: str = "legacy_bridge_mission",
        world_state: dict[str, Any] | None = None,
    ) -> LegacyExecutionPlan:
        """Build a legacy-compatible mode-switch plan from natural language."""
        graph = self.llm_interface.parse_command(
            command=command,
            mission_id=mission_id,
            world_state=world_state,
        )
        return LegacyExecutionPlan(
            mission_id=mission_id,
            command=command,
            mode_sequence=self.graph_to_mode_sequence(graph),
            skill_sequence=[subtask.skill for subtask in graph.subtasks],
            subtask_ids=[subtask.id for subtask in graph.subtasks],
            graph=graph,
            runtime_layout=self.runtime_layout,
        )
