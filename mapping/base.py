"""
Base Occupancy Grid Interface
Defines the standard abstract contract for spatial mapping and safety margins.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from utils.data_types import Pose2D, ObstacleDetection, GridMap


class BaseOccupancyGrid(ABC):
    """Abstract base class for 2D costmaps and occupancy grid mapping."""

    @abstractmethod
    def reset(self) -> None:
        """Clears the map to initial unvisited state."""
        pass

    @abstractmethod
    def update(self, robot_pose: Pose2D, detection: ObstacleDetection) -> None:
        """
        Fuses new local obstacle observations into the global grid map.
        :param robot_pose: Estimated or ground-truth pose of UGV.
        :param detection: Obstacle points in robot local frame.
        """
        pass

    @abstractmethod
    def get_grid(self) -> GridMap:
        """Returns the raw un-inflated occupancy grid."""
        pass

    @abstractmethod
    def get_inflated_grid(self) -> GridMap:
        """Returns the safety-inflated costmap used by path planners."""
        pass

    @abstractmethod
    def is_occupied(self, col: int, row: int, use_inflation: bool = True) -> bool:
        """Checks if a given grid cell is an obstacle or within the inflation safety zone."""
        pass
