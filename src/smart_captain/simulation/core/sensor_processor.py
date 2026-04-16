"""Reusable sensor processing utilities for simulation adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from skimage.measure import block_reduce


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Normalize angles to `[-pi, pi)`."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def deg_to_rad(angle_deg: np.ndarray | float) -> np.ndarray | float:
    """Convert degrees to radians."""
    return np.asarray(angle_deg) / 180.0 * np.pi


@dataclass
class RangeFinderGrid:
    """Structured view over the 9x24 range-finder layout used today."""

    blocksize_reduce: int = 3
    vertical_angles_deg: tuple[int, ...] = (-60, -45, -30, -15, 0, 15, 30, 45, 60)
    horizontal_angles_deg: tuple[int, ...] = (
        0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
        180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345,
    )
    max_dist: float = 10.0
    sensor_key_by_angle_deg: Dict[int, str] = field(default_factory=lambda: {
        -60: "RangeFinderSensor8",
        -45: "RangeFinderSensor7",
        -30: "RangeFinderSensor6",
        -15: "RangeFinderSensor5",
        0: "RangeFinderSensor0",
        15: "RangeFinderSensor1",
        30: "RangeFinderSensor2",
        45: "RangeFinderSensor3",
        60: "RangeFinderSensor4",
    })

    def __post_init__(self) -> None:
        self.vertical_angles = np.asarray(self.vertical_angles_deg, dtype=np.float32) * np.pi / 180.0
        horizontal = np.asarray(self.horizontal_angles_deg, dtype=np.float32)
        self.horizontal_angles = wrap_to_pi(horizontal * np.pi / 180.0)
        self.n_vertical = len(self.vertical_angles_deg)
        self.n_horizontal = len(self.horizontal_angles_deg)
        self.n_rays = self.n_vertical * self.n_horizontal
        self.alpha = np.repeat(self.vertical_angles, self.n_horizontal)
        self.beta = np.tile(self.horizontal_angles, self.n_vertical)
        self.alpha_max = np.pi / 3
        self.beta_max = np.pi
        self.intersection_distances = np.zeros(self.n_rays, dtype=np.float32)

    def update_from_sensor_return(self, sensor_return: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten 9 range-finder rows into a single ray-distance vector."""
        dist_matrix = np.vstack([
            sensor_return[self.sensor_key_by_angle_deg[angle]]
            for angle in self.vertical_angles_deg
        ])
        self.intersection_distances = dist_matrix.astype(np.float32).flatten()
        return self.intersection_distances

    @property
    def distance_matrix(self) -> np.ndarray:
        """Return the current `9x24` distance matrix."""
        return self.intersection_distances.reshape((self.n_vertical, self.n_horizontal))

    @property
    def reduced_distances(self) -> np.ndarray:
        """Return block-reduced ray distances."""
        reduced = block_reduce(
            self.distance_matrix,
            block_size=(self.blocksize_reduce, self.blocksize_reduce),
            func=np.median,
        )
        return reduced.flatten()
