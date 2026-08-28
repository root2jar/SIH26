"""
Simulation World Environment
Generates outdoor terrain layouts with circular rocks, trees, and boundary barriers.
"""

from typing import List, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class WorldObstacle:
    """Represents a physical object in the simulation world."""
    x: float
    y: float
    radius: float
    color_rgb: Tuple[int, int, int] = (50, 45, 40)  # Visual color
    obstacle_type: str = "rock"                     # 'rock', 'boulder', 'bush', 'barrier'
    is_dynamic: bool = False
    vx: float = 0.0
    vy: float = 0.0


class SimulationWorld:
    """
    Outdoor navigation environment with grass ground, dirt tracks, and obstacles.
    """

    def __init__(self, width: float = 24.0, height: float = 18.0):
        self.width = width
        self.height = height
        self.obstacles: List[WorldObstacle] = []
        self.start_pose: Tuple[float, float, float] = (2.5, 2.5, 0.0)
        self.goal_pos: Tuple[float, float] = (21.0, 15.0)

        self.setup_default_scenario()

    def setup_default_scenario(self) -> None:
        """Sets up a realistic outdoor obstacle course with varied materials & shapes."""
        self.obstacles.clear()

        # Varied outdoor obstacle field: (x, y, radius, color_rgb, type)
        preset_obstacles = [
            (6.0, 5.0, 1.2, (95, 100, 105), "rock"),      # Gray granite rock
            (7.5, 9.5, 1.4, (45, 75, 35), "bush"),        # Dense foliage bush
            (11.0, 4.0, 1.0, (50, 45, 42), "boulder"),    # Dark boulder
            (12.0, 13.0, 1.5, (85, 55, 35), "barrier"),   # Timber barrier
            (14.5, 8.5, 1.3, (105, 110, 115), "rock"),    # Granite stone
            (17.0, 5.5, 1.1, (40, 68, 30), "bush"),       # Olive bush
            (18.5, 12.0, 1.2, (55, 50, 46), "boulder"),   # Basalt rock
            (9.0, 14.5, 1.0, (110, 85, 55), "rock"),      # Sandstone
            (15.0, 15.5, 0.9, (45, 70, 32), "bush"),      # Shrub
            (4.0, 11.0, 1.1, (80, 50, 30), "barrier"),    # Wooden post
            (19.0, 8.5, 1.0, (90, 95, 100), "rock"),      # Slate rock
            (8.5, 2.5, 0.9, (52, 48, 44), "boulder"),     # Dark stone
        ]

        for x, y, r, color, o_type in preset_obstacles:
            self.obstacles.append(
                WorldObstacle(x=x, y=y, radius=r, color_rgb=color, obstacle_type=o_type)
            )

    def add_random_obstacles(self, count: int = 10, min_radius: float = 0.6, max_radius: float = 1.4) -> None:
        """Adds randomly scattered obstacles avoiding start and goal areas."""
        np.random.seed(42)
        added = 0
        while added < count:
            x = np.random.uniform(3.0, self.width - 3.0)
            y = np.random.uniform(3.0, self.height - 3.0)
            r = np.random.uniform(min_radius, max_radius)

            # Avoid placing directly on start or goal
            dist_start = np.hypot(x - self.start_pose[0], y - self.start_pose[1])
            dist_goal = np.hypot(x - self.goal_pos[0], y - self.goal_pos[1])

            if dist_start > 2.5 and dist_goal > 2.5:
                self.obstacles.append(WorldObstacle(x=x, y=y, radius=r))
                added += 1

    def update_dynamic_obstacles(self, dt: float) -> None:
        """Updates moving obstacles if any exist."""
        for obs in self.obstacles:
            if obs.is_dynamic:
                obs.x += obs.vx * dt
                obs.y += obs.vy * dt
                # Bounce off boundaries
                if obs.x < 1.0 or obs.x > self.width - 1.0:
                    obs.vx *= -1
                if obs.y < 1.0 or obs.y > self.height - 1.0:
                    obs.vy *= -1
