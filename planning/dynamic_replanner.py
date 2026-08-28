"""
Dynamic Replanning & Collision Avoidance
Monitors active trajectory against newly discovered obstacles and triggers
local or global A* replanning when impending collisions are detected.
"""

import math
from typing import Tuple, Optional
from planning.base import BasePlanner
from utils.data_types import Pose2D, GridMap, Path2D
from utils.math_helpers import world_to_grid, is_inside_grid, euclidean_distance


class DynamicReplanner:
    """
    Continuous collision monitor and replan coordinator.
    """

    def __init__(
        self,
        planner: BasePlanner,
        collision_check_horizon: float = 3.5,  # Distance ahead to check along path (meters)
        occupied_threshold: int = 65,
        min_replan_interval: float = 0.25,     # Minimum seconds between replans to prevent chattering
    ):
        self.planner = planner
        self.collision_check_horizon = collision_check_horizon
        self.occupied_threshold = occupied_threshold
        self.min_replan_interval = min_replan_interval
        self.last_replan_time: float = -1.0
        self.active_path = Path2D()
        self.current_goal: Optional[Tuple[float, float]] = None

    def set_goal(self, start_pose: Pose2D, goal_world: Tuple[float, float], grid_map: GridMap) -> Path2D:
        """Initializes navigation to a new goal point."""
        self.current_goal = goal_world
        self.active_path = self.planner.plan(
            start_world=start_pose.position_tuple(),
            goal_world=goal_world,
            grid_map=grid_map,
        )
        self.last_replan_time = 0.0
        return self.active_path

    def update(self, current_pose: Pose2D, grid_map: GridMap, timestamp: float = 0.0) -> Tuple[Path2D, bool]:
        """
        Validates the active path. If an obstacle blocks the path ahead, recalculates route.
        :return: (active_path, replanned_flag)
        """
        if self.current_goal is None:
            return self.active_path, False

        # If no valid path exists currently, attempt to find one
        if not self.active_path.is_valid or self.active_path.is_empty():
            self.active_path = self.planner.plan(
                start_world=current_pose.position_tuple(),
                goal_world=self.current_goal,
                grid_map=grid_map,
            )
            self.last_replan_time = timestamp
            return self.active_path, True

        # Check for upcoming collisions along waypoints
        blocked, min_dist = self._check_path_blocked_with_dist(current_pose, grid_map)

        if blocked:
            # Replan if cooldown period elapsed OR imminent collision on path (< 0.6m)
            if (timestamp - self.last_replan_time >= self.min_replan_interval) or (min_dist < 0.6):
                new_path = self.planner.plan(
                    start_world=current_pose.position_tuple(),
                    goal_world=self.current_goal,
                    grid_map=grid_map,
                )
                if new_path.is_valid and not new_path.is_empty():
                    self.active_path = new_path
                    self.last_replan_time = timestamp
                    return self.active_path, True

        return self.active_path, False

    def _is_path_blocked(self, current_pose: Pose2D, grid_map: GridMap) -> bool:
        blocked, _ = self._check_path_blocked_with_dist(current_pose, grid_map)
        return blocked

    def _check_path_blocked_with_dist(self, current_pose: Pose2D, grid_map: GridMap) -> Tuple[bool, float]:
        """
        Continuously checks if upcoming trajectory segments within the collision horizon
        intersect newly discovered obstacles in the costmap. Returns (is_blocked, min_distance).
        """
        if self.active_path.is_empty() or not self.active_path.is_valid:
            return True, 0.0

        cur_pos = current_pose.position_tuple()
        points = [cur_pos] + self.active_path.waypoints

        min_dist = float("inf")
        is_blocked = False

        # Step through all line segments along the active trajectory
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            seg_dist = euclidean_distance(p1, p2)

            step_size = max(0.05, grid_map.resolution * 0.5)
            num_steps = max(1, int(math.ceil(seg_dist / step_size)))

            for s in range(num_steps + 1):
                t = s / float(num_steps)
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])

                check_dist = euclidean_distance(cur_pos, (x, y))
                if check_dist > self.collision_check_horizon:
                    return is_blocked, min_dist

                col, row = world_to_grid(x, y, grid_map)
                if is_inside_grid(col, row, grid_map):
                    if grid_map.data[row, col] >= self.occupied_threshold:
                        is_blocked = True
                        if check_dist < min_dist:
                            min_dist = check_dist

        return is_blocked, min_dist

