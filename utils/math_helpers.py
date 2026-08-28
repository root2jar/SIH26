"""
Mathematical & Coordinate Transformation Utilities
Handles transformations across World, Robot Body, Camera, and Grid coordinates.
"""

import math
from typing import Tuple
import numpy as np
from utils.data_types import Pose2D, GridMap


def normalize_angle(angle: float) -> float:
    """Wraps an angle into standard [-pi, pi] range."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculates 2D Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def robot_to_world(local_point: Tuple[float, float], robot_pose: Pose2D) -> Tuple[float, float]:
    """
    Transforms a 2D point from UGV local body frame to global World frame.
    In local body frame:
      +x is forward along heading theta
      +y is left (perpendicular)
    """
    lx, ly = local_point
    cos_t = math.cos(robot_pose.theta)
    sin_t = math.sin(robot_pose.theta)

    world_x = robot_pose.x + (lx * cos_t - ly * sin_t)
    world_y = robot_pose.y + (lx * sin_t + ly * cos_t)
    return (world_x, world_y)


def world_to_robot(world_point: Tuple[float, float], robot_pose: Pose2D) -> Tuple[float, float]:
    """Transforms a 2D point from global World frame to UGV local body frame."""
    dx = world_point[0] - robot_pose.x
    dy = world_point[1] - robot_pose.y
    cos_t = math.cos(robot_pose.theta)
    sin_t = math.sin(robot_pose.theta)

    local_x = dx * cos_t + dy * sin_t
    local_y = -dx * sin_t + dy * cos_t
    return (local_x, local_y)


def world_to_grid(world_x: float, world_y: float, grid_map: GridMap) -> Tuple[int, int]:
    """
    Converts continuous world coordinates (meters) to discrete grid cell indices (col, row).
    Returns (col, row) where col is along X (width) and row is along Y (height).
    """
    col = int((world_x - grid_map.origin_x) / grid_map.resolution)
    row = int((world_y - grid_map.origin_y) / grid_map.resolution)
    return (col, row)


def grid_to_world(col: int, row: int, grid_map: GridMap) -> Tuple[float, float]:
    """
    Converts discrete grid cell indices (col, row) to continuous world coordinates (meters).
    Returns cell center (world_x, world_y).
    """
    world_x = grid_map.origin_x + (col + 0.5) * grid_map.resolution
    world_y = grid_map.origin_y + (row + 0.5) * grid_map.resolution
    return (world_x, world_y)


def is_inside_grid(col: int, row: int, grid_map: GridMap) -> bool:
    """Checks whether the given (col, row) lies within valid grid boundaries."""
    return 0 <= col < grid_map.width_cells and 0 <= row < grid_map.height_cells
