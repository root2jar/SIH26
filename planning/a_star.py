"""
A* (A-Star) 2D Grid Path Planner
Finds optimal shortest path from current UGV position to goal waypoint
on the inflated occupancy grid while preferring paths with higher clearance.
"""

import heapq
import math
from typing import Tuple, List, Dict, Optional
import numpy as np
from planning.base import BasePlanner
from utils.data_types import GridMap, Path2D
from utils.math_helpers import world_to_grid, grid_to_world, is_inside_grid, euclidean_distance


class AStarPlanner(BasePlanner):
    """
    8-connected A* search algorithm operating on the 2D Occupancy Grid.
    """

    def __init__(
        self,
        heuristic_weight: float = 1.0,
        allow_diagonal: bool = True,
        occupied_threshold: int = 65,
        safety_penalty_weight: float = 0.5,
    ):
        self.heuristic_weight = heuristic_weight
        self.allow_diagonal = allow_diagonal
        self.occupied_threshold = occupied_threshold
        self.safety_penalty_weight = safety_penalty_weight

        # 8-connected motion primitives: (d_col, d_row, step_cost)
        if self.allow_diagonal:
            self.motions = [
                (1, 0, 1.0),
                (-1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
                (1, 1, 1.414),
                (-1, 1, 1.414),
                (1, -1, 1.414),
                (-1, -1, 1.414),
            ]
        else:
            self.motions = [
                (1, 0, 1.0),
                (-1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
            ]

    def plan(
        self,
        start_world: Tuple[float, float],
        goal_world: Tuple[float, float],
        grid_map: GridMap,
    ) -> Path2D:
        """
        Executes A* search on grid_map from start_world to goal_world.
        """
        start_cell = world_to_grid(start_world[0], start_world[1], grid_map)
        goal_cell = world_to_grid(goal_world[0], goal_world[1], grid_map)

        # Validate start and goal are inside grid
        if not is_inside_grid(start_cell[0], start_cell[1], grid_map):
            return Path2D(waypoints=[], is_valid=False)

        if not is_inside_grid(goal_cell[0], goal_cell[1], grid_map):
            # Clamp goal to closest valid boundary
            goal_cell = (
                max(0, min(goal_cell[0], grid_map.width_cells - 1)),
                max(0, min(goal_cell[1], grid_map.height_cells - 1)),
            )

        # If start or goal is in an obstacle, find nearest free neighbor cell
        start_cell = self._find_nearest_free_cell(start_cell, grid_map)
        goal_cell = self._find_nearest_free_cell(goal_cell, grid_map)

        if start_cell is None or goal_cell is None:
            return Path2D(waypoints=[], is_valid=False)

        # Priority queue entries: (f_score, g_score, (col, row))
        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_cell: 0.0}

        h_start = self._heuristic(start_cell, goal_cell)
        heapq.heappush(open_set, (h_start, 0.0, start_cell))
        visited = set()

        found_goal = False

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == goal_cell:
                found_goal = True
                break

            if current in visited:
                continue
            visited.add(current)

            col, row = current

            for dc, dr, move_cost in self.motions:
                n_col = col + dc
                n_row = row + dr
                neighbor = (n_col, n_row)

                if not is_inside_grid(n_col, n_row, grid_map):
                    continue

                cell_cost = grid_map.data[n_row, n_col]
                # Treat cells above occupied threshold as impassable
                if cell_cost >= self.occupied_threshold:
                    continue

                # Add inflation penalty cost to favor open space clearance
                safety_penalty = (cell_cost / 100.0) * self.safety_penalty_weight
                tentative_g = current_g + move_cost + safety_penalty

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic_weight * self._heuristic(neighbor, goal_cell)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        if not found_goal:
            return Path2D(waypoints=[], is_valid=False)

        # Reconstruct path backwards from goal to start
        raw_path_cells: List[Tuple[int, int]] = []
        curr = goal_cell
        while curr in came_from:
            raw_path_cells.append(curr)
            curr = came_from[curr]
        raw_path_cells.append(start_cell)
        raw_path_cells.reverse()

        # Convert grid cells to continuous world coordinates (meters)
        world_waypoints = [grid_to_world(c[0], c[1], grid_map) for c in raw_path_cells]

        # Smooth / shortcut path using line-of-sight checks
        smoothed_waypoints = self._smooth_path(world_waypoints, grid_map)

        return Path2D(waypoints=smoothed_waypoints, is_valid=True)

    def _heuristic(self, cell1: Tuple[int, int], cell2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return math.hypot(cell2[0] - cell1[0], cell2[1] - cell1[1])

    def _find_nearest_free_cell(self, cell: Tuple[int, int], grid_map: GridMap) -> Optional[Tuple[int, int]]:
        """Finds closest free cell if the given cell is occupied."""
        if grid_map.data[cell[1], cell[0]] < self.occupied_threshold:
            return cell

        col, row = cell
        for radius in range(1, 8):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nc, nr = col + dc, row + dr
                    if is_inside_grid(nc, nr, grid_map):
                        if grid_map.data[nr, nc] < self.occupied_threshold:
                            return (nc, nr)
        return None

    def _smooth_path(self, waypoints: List[Tuple[float, float]], grid_map: GridMap) -> List[Tuple[float, float]]:
        """Reduces unnecessary zig-zag waypoints using line-of-sight shortcutting."""
        if len(waypoints) <= 2:
            return waypoints

        smoothed = [waypoints[0]]
        curr_idx = 0

        while curr_idx < len(waypoints) - 1:
            furthest_idx = len(waypoints) - 1
            for target_idx in range(len(waypoints) - 1, curr_idx, -1):
                if self._has_line_of_sight(waypoints[curr_idx], waypoints[target_idx], grid_map):
                    furthest_idx = target_idx
                    break

            smoothed.append(waypoints[furthest_idx])
            curr_idx = furthest_idx

        return smoothed

    def _has_line_of_sight(self, p1: Tuple[float, float], p2: Tuple[float, float], grid_map: GridMap) -> bool:
        """Checks if a straight line between two world points is free of obstacles."""
        dist = euclidean_distance(p1, p2)
        steps = max(2, int(dist / (grid_map.resolution * 0.5)))
        for i in range(steps + 1):
            t = i / float(steps)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            c, r = world_to_grid(x, y, grid_map)
            if not is_inside_grid(c, r, grid_map) or grid_map.data[r, c] >= self.occupied_threshold:
                return False
        return True
