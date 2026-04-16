"""Pier Harbor scenario spec."""

from __future__ import annotations

from smart_captain.simulation.core.scenario_manager import ScenarioSpec


class PierHarborScenario:
    """Default Pier Harbor setup matching the current hovering environment."""

    @staticmethod
    def default() -> ScenarioSpec:
        return ScenarioSpec(
            name="Hovering",
            world="PierHarbor",
            package_name="Ocean",
            main_agent="auv0",
            ticks_per_sec=200,
            agents=[
                {
                    "agent_name": "auv0",
                    "agent_type": "HoveringAUV",
                    "sensors": [
                        {
                            "sensor_type": "DynamicsSensor",
                            "socket": "COM",
                            "configuration": {
                                "UseCOM": True,
                                "UseRPY": True,
                            },
                        },
                        {"sensor_type": "VelocitySensor", "socket": "IMUSocket"},
                    ],
                    "control_scheme": 0,
                    "location": [0, 0, -10],
                    "rotation": [0.0, 0.0, 0.0],
                }
            ],
        )
