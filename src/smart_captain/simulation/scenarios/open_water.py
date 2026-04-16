"""Open water scenario spec placeholder."""

from __future__ import annotations

from smart_captain.simulation.core.scenario_manager import ScenarioSpec


class OpenWaterScenario:
    """Reserved landing spot for open-water obstacle avoidance scenarios."""

    @staticmethod
    def default() -> ScenarioSpec:
        return ScenarioSpec(
            name="OpenWaterMission",
            world="OpenWater",
            package_name="Ocean",
            main_agent="auv0",
            ticks_per_sec=200,
            agents=[],
        )
