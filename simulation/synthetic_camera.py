"""
Synthetic Camera Vision Simulator
Renders a realistic forward-looking optical camera view and depth map
from the UGV's true pose in the simulation world.
"""

import math
import cv2
import numpy as np
from simulation.world import SimulationWorld
from utils.data_types import Pose2D, CameraFrame
from utils.math_helpers import world_to_robot, normalize_angle


class SyntheticCamera:
    """
    Renders the first-person perspective seen by an optical camera mounted on the front of the UGV.
    """

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        fov_deg: float = 70.0,
        max_range: float = 8.0,
        camera_height: float = 0.45,
    ):
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.fov_rad = math.radians(fov_deg)
        self.half_fov = self.fov_rad / 2.0
        self.max_range = max_range
        self.camera_height = camera_height

        # Pre-generate outdoor ground texture (dirt & grass noise for visual odometry tracking)
        np.random.seed(100)
        self.ground_texture_base = np.random.randint(40, 75, (height // 2, width), dtype=np.uint8)

    def render(self, robot_pose: Pose2D, world: SimulationWorld, timestamp: float = 0.0) -> CameraFrame:
        """
        Renders first-person RGB viewport and depth buffer from robot pose.
        """
        # Create blank image buffer (H, W, 3)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.full((self.height, self.width), self.max_range, dtype=np.float32)

        horizon_y = self.height // 2

        # 1. Sky & Distant Horizon Rendering (Sky blue gradient to pale atmospheric haze)
        for y in range(horizon_y):
            ratio = y / float(horizon_y)
            r = int(135 + ratio * 45)
            g = int(175 + ratio * 35)
            b = int(220 - ratio * 10)
            img[y, :] = (r, g, b)

        # Draw subtle distant outdoor hill contours along horizon
        for x in range(self.width):
            hill_offset = int(math.sin(x * 0.04) * 4 + math.sin(x * 0.015) * 6)
            hy = max(10, horizon_y - 8 + hill_offset)
            img[hy:horizon_y, x] = (100, 135, 95)  # Distant forest green hills

        # 2. Outdoor Ground Plane Terrain (Vibrant grassy green with dirt texture)
        for y in range(horizon_y, self.height):
            ratio = (y - horizon_y) / float(self.height - horizon_y)  # 0 at horizon, 1 at bumper
            dist_approx = self.camera_height / max(0.05, ratio)

            # Earth tone: Grassy green with soil undertones
            r_base = int(70 + ratio * 35)
            g_base = int(140 + ratio * 35)
            b_base = int(45 + ratio * 20)

            # Apply optical texture noise for computer vision & feature tracking
            noise = self.ground_texture_base[y - horizon_y, :]
            img[y, :, 0] = np.clip(r_base + (noise - 50) // 2, 0, 255)
            img[y, :, 1] = np.clip(g_base + (noise - 50), 0, 255)
            img[y, :, 2] = np.clip(b_base + (noise - 50) // 3, 0, 255)

            depth[y, :] = min(self.max_range, dist_approx)

        # 3. Obstacle Projection & Shading
        visible_obstacles = []
        for obs in world.obstacles:
            lx, ly = world_to_robot((obs.x, obs.y), robot_pose)

            if lx <= 0.3:  # Behind camera
                continue

            dist = math.hypot(lx, ly)
            if dist > self.max_range + obs.radius:
                continue

            angle = math.atan2(ly, lx)

            if abs(angle) <= self.half_fov + math.atan2(obs.radius, dist):
                visible_obstacles.append((dist, lx, ly, angle, obs))

        # Sort furthest first (Painter's algorithm)
        visible_obstacles.sort(key=lambda item: item[0], reverse=True)

        for dist, lx, ly, angle, obs in visible_obstacles:
            # Perspective screen position
            u_center = int(self.width / 2.0 - (angle / self.half_fov) * (self.width / 2.0))
            apparent_width = int((obs.radius * 2.0 / dist) * (self.width / (2.0 * math.tan(self.half_fov))))
            apparent_height = int((obs.radius * 2.2 / dist) * (self.height / 1.6))

            # Ground contact line
            ground_v_ratio = self.camera_height / max(0.15, dist)
            v_bottom = int(horizon_y + ground_v_ratio * (self.height - horizon_y))
            v_top = max(10, v_bottom - apparent_height)

            if apparent_width < 6 or apparent_height < 6:
                continue

            x1 = max(0, u_center - apparent_width // 2)
            x2 = min(self.width, u_center + apparent_width // 2)
            y1 = max(0, v_top)
            y2 = min(self.height, v_bottom)

            if x1 >= x2 or y1 >= y2:
                continue

            # Drop shadow on the ground beneath obstacle
            shadow_h = max(2, apparent_height // 6)
            cv2.ellipse(
                img,
                (u_center, min(self.height - 2, v_bottom)),
                (apparent_width // 2, shadow_h),
                0, 0, 360,
                (25, 45, 20),
                -1,
            )

            color = obs.color_rgb

            # Render based on obstacle material / type
            if getattr(obs, "obstacle_type", "rock") == "bush":
                # Organic bumpy bush with leaf clusters
                cv2.ellipse(img, (u_center, (y1 + y2) // 2), (apparent_width // 2, apparent_height // 2), 0, 0, 360, color, -1)
                cv2.ellipse(img, (u_center, (y1 + y2) // 2), (apparent_width // 2, apparent_height // 2), 0, 0, 360, (20, 35, 15), 2)
                # Foliage texture bumps
                for bump_dx in [-apparent_width // 4, 0, apparent_width // 4]:
                    cv2.circle(img, (u_center + bump_dx, y1 + apparent_height // 3), max(2, apparent_height // 4), (color[0] + 15, color[1] + 15, color[2] + 10), -1)
            elif getattr(obs, "obstacle_type", "rock") == "barrier":
                # Wooden barrier with timber texture
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (25, 15, 10), 2)
                # Wood grain horizontal planks
                for plank_y in range(y1, y2, max(4, apparent_height // 3)):
                    cv2.line(img, (x1, plank_y), (x2, plank_y), (max(0, color[0] - 25), max(0, color[1] - 25), max(0, color[2] - 20)), 1)
            else:
                # Faceted Rock / Boulder with shadow & highlight
                pts = np.array([
                    [x1 + apparent_width // 5, y2],
                    [x1, y1 + apparent_height // 3],
                    [x1 + apparent_width // 4, y1],
                    [x2 - apparent_width // 4, y1],
                    [x2, y1 + apparent_height // 3],
                    [x2 - apparent_width // 6, y2],
                ], dtype=np.int32)
                cv2.fillPoly(img, [pts], color)
                cv2.polylines(img, [pts], True, (25, 25, 25), 2)
                # Top specular rock highlight
                cv2.line(img, (x1 + apparent_width // 4, y1 + 2), (x2 - apparent_width // 4, y1 + 2), (min(255, color[0] + 40), min(255, color[1] + 40), min(255, color[2] + 40)), 2)

            # Update depth buffer
            obs_depth = max(0.3, dist - obs.radius)
            depth[y1:y2, x1:x2] = np.minimum(depth[y1:y2, x1:x2], obs_depth)

        return CameraFrame(
            image=img,
            depth=depth,
            timestamp=timestamp,
            fov_deg=self.fov_deg,
        )
