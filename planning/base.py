"""
Base Path Planner Interface
Defines the standard abstract contract for global and local path planners.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from utils.data_types import GridMap, Path2D


class BasePlanner(ABC):
    """Abstract interface for path planning algorithms (A*, Dijkstra, RRT, etc.)."""

    @abstractmethod
    def plan(
        self,
        start_world: Tuple[float, float],
        goal_world: Tuple[float, float],
        grid_map: GridMap,
    ) -> Path2D:
        """
        Computes a collision-free path from start to goal coordinates.
        :param start_world: (x, y) world position of start.
        :param goal_world: (x, y) world position of destination.
        :param grid_map: Occupancy grid (preferably inflated for safety).
        :return: Path2D containing sequence of (x, y) world waypoints.
        """
        pass
