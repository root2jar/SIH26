"""
2D Occupancy Grid Map Implementation
Fuses robot pose estimates with perceived obstacles to maintain a global costmap.
"""

import math
from typing import Tuple, List
import numpy as np
from mapping.base import BaseOccupancyGrid
from mapping.inflation import ObstacleInflationLayer
from utils.data_types import Pose2D, ObstacleDetection, GridMap
from utils.math_helpers import robot_to_world, world_to_grid, is_inside_grid


class OccupancyGridMap(BaseOccupancyGrid):
    """
    Maintains a 2D discrete grid map of the outdoor environment.
    Values: 0 (free space), 100 (lethal obstacle), intermediate (inflation cost).
    """

    def __init__(
        self,
        width_meters: float = 24.0,
        height_meters: float = 18.0,
        resolution: float = 0.15,
        inflation_radius: float = 0.5,
    ):
        self.width_meters = width_meters
        self.height_meters = height_meters
        self.resolution = resolution

        self.width_cells = int(math.ceil(width_meters / resolution))
        self.height_cells = int(math.ceil(height_meters / resolution))

        self.raw_data = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
        self.inflator = ObstacleInflationLayer(resolution=resolution, inflation_radius=inflation_radius)
        self.inflated_data = np.zeros_like(self.raw_data)

        self._grid_cache = GridMap(
            data=self.raw_data,
            resolution=self.resolution,
            origin_x=0.0,
            origin_y=0.0,
        )

    def reset(self) -> None:
        self.raw_data.fill(0)
        self.inflated_data.fill(0)

    def update(self, robot_pose: Pose2D, detection: ObstacleDetection) -> None:
        """
        Integrates visual obstacle detections into the 2D occupancy grid and regenerates the inflated costmap.
        """
        updated = False
        robot_col, robot_row = world_to_grid(robot_pose.x, robot_pose.y, self._grid_cache)

        if hasattr(detection, "obstacle_regions") and detection.obstacle_regions:
            for reg in detection.obstacle_regions:
                # Obstacle center in world coordinates
                world_cx, world_cy = robot_to_world((reg.local_x, reg.local_y), robot_pose)
                c_col, c_row = world_to_grid(world_cx, world_cy, self._grid_cache)
                r_cells = max(1, int(math.ceil(reg.radius_meters / self.resolution)))

                # Mark circular obstacle footprint
                for dr in range(-r_cells, r_cells + 1):
                    for dc in range(-r_cells, r_cells + 1):
                        if dr * dr + dc * dc <= r_cells * r_cells:
                            col = c_col + dc
                            row = c_row + dr
                            if is_inside_grid(col, row, self._grid_cache):
                                if self.raw_data[row, col] != 100:
                                    self.raw_data[row, col] = 100
                                    updated = True

                # Clear line of sight from robot to front edge of obstacle
                angle = math.atan2(reg.local_y, reg.local_x)
                front_local = (reg.distance_meters * math.cos(angle), reg.distance_meters * math.sin(angle))
                front_wx, front_wy = robot_to_world(front_local, robot_pose)
                f_col, f_row = world_to_grid(front_wx, front_wy, self._grid_cache)
                self._trace_free_line(robot_col, robot_row, f_col, f_row)

        elif detection.local_points:
            r_cells = max(1, int(math.ceil(0.5 / self.resolution)))
            for local_pt in detection.local_points:
                world_pt = robot_to_world(local_pt, robot_pose)
                c_col, c_row = world_to_grid(world_pt[0], world_pt[1], self._grid_cache)

                for dr in range(-r_cells, r_cells + 1):
                    for dc in range(-r_cells, r_cells + 1):
                        if dr * dr + dc * dc <= r_cells * r_cells:
                            col = c_col + dc
                            row = c_row + dr
                            if is_inside_grid(col, row, self._grid_cache):
                                if self.raw_data[row, col] != 100:
                                    self.raw_data[row, col] = 100
                                    updated = True

                self._trace_free_line(robot_col, robot_row, c_col, c_row)

        if updated or detection.local_points:
            self.inflated_data = self.inflator.inflate(self.raw_data)

    def add_static_obstacle_circle(self, center_x: float, center_y: float, radius: float) -> None:
        """Helper to add known static circular obstacles to the map (trees, boulders)."""
        c_col, c_row = world_to_grid(center_x, center_y, self._grid_cache)
        r_cells = int(math.ceil(radius / self.resolution))

        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dr * dr + dc * dc <= r_cells * r_cells:
                    col = c_col + dc
                    row = c_row + dr
                    if is_inside_grid(col, row, self._grid_cache):
                        self.raw_data[row, col] = 100

        self.inflated_data = self.inflator.inflate(self.raw_data)

    def add_boundary_walls(self, thickness_meters: float = 0.4) -> None:
        """Adds boundary wall obstacles around the map perimeter."""
        t_cells = max(1, int(math.ceil(thickness_meters / self.resolution)))
        self.raw_data[:t_cells, :] = 100
        self.raw_data[-t_cells:, :] = 100
        self.raw_data[:, :t_cells] = 100
        self.raw_data[:, -t_cells:] = 100
        self.inflated_data = self.inflator.inflate(self.raw_data)

    def get_grid(self) -> GridMap:
        self._grid_cache.data = self.raw_data
        return self._grid_cache

    def get_inflated_grid(self) -> GridMap:
        return GridMap(
            data=self.inflated_data,
            resolution=self.resolution,
            origin_x=0.0,
            origin_y=0.0,
        )

    def is_occupied(self, col: int, row: int, use_inflation: bool = True) -> bool:
        if not is_inside_grid(col, row, self._grid_cache):
            return True
        cost = self.inflated_data[row, col] if use_inflation else self.raw_data[row, col]
        return cost >= 65

    def _trace_free_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Bresenham line algorithm to clear free cells along raycast (excluding terminal obstacle)."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        curr_x, curr_y = x0, y0
        while (curr_x != x1 or curr_y != y1):
            if is_inside_grid(curr_x, curr_y, self._grid_cache):
                if self.raw_data[curr_y, curr_x] != 100:
                    self.raw_data[curr_y, curr_x] = 0
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += sx
            if e2 < dx:
                err += dx
                curr_y += sy
