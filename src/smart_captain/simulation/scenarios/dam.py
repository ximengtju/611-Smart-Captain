"""Dam scenario spec placeholder."""

from __future__ import annotations

from smart_captain.simulation.core.scenario_manager import ScenarioSpec


class DamScenario:
    """Reserved landing spot for the dam inspection scenario."""

    @staticmethod
    def default() -> ScenarioSpec:
        return ScenarioSpec(
            name="DamMission",
            world="Dam",
            package_name="Ocean",
            main_agent="auv0",
            ticks_per_sec=200,
            agents=[],
        )
