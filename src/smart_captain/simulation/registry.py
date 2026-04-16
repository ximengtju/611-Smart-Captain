"""Simulation registries.

These registries are intentionally lightweight placeholders so the future
framework can add scenarios and sensors without editing the main entrypoint.
"""

SCENARIO_REGISTRY = {
    "pier_harbor": "smart_captain.simulation.scenarios.pier_harbor:PierHarborScenario",
    "dam": "smart_captain.simulation.scenarios.dam:DamScenario",
    "open_water": "smart_captain.simulation.scenarios.open_water:OpenWaterScenario",
}

SENSOR_REGISTRY = {
    "radar": "smart_captain.simulation.sensors.radar:RadarSensorAdapter",
    "imaging_sonar": "smart_captain.simulation.sensors.imaging_sonar:ImagingSonarAdapter",
    "rgb_camera": "smart_captain.simulation.sensors.rgb_camera:RGBCameraAdapter",
    "bst": "smart_captain.simulation.sensors.bst:BSTSensorAdapter",
}
