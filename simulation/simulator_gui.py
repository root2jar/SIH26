"""
Pygame Multi-Panel Simulation Dashboard
Displays the 2D Top-Down World, 4-Stage Classical CV Perception Pipeline,
Live Reconstructed Occupancy Grid / Costmap, and Real-Time Telemetry.
"""

import math
from typing import Tuple, List, Optional
import pygame
import numpy as np
import cv2

from simulation.world import SimulationWorld
from simulation.robot_sim import RobotSim
from utils.data_types import Pose2D, CameraFrame, GridMap, Path2D, TwistCommand, ObstacleDetection


class SimulatorGUI:
    """
    Interactive visualization dashboard built with Pygame.
    """

    def __init__(
        self,
        screen_width: int = 1400,
        screen_height: int = 820,
        world_width: float = 24.0,
        world_height: float = 18.0,
    ):
        pygame.init()
        pygame.font.init()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.world_width = world_width
        self.world_height = world_height

        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("SIH 2026: Vision-Based Autonomous UGV Navigation Prototype")
        self.clock = pygame.time.Clock()

        # Layout geometry
        self.world_view_rect = pygame.Rect(10, 10, 680, 800)
        self.right_panel_x = 700

        # Coordinate scaling (meters -> pixels)
        self.scale_x = (self.world_view_rect.width - 20) / self.world_width
        self.scale_y = (self.world_view_rect.height - 20) / self.world_height
        self.scale = min(self.scale_x, self.scale_y)
        self.offset_x = self.world_view_rect.x + 10
        self.offset_y = self.world_view_rect.y + 10

        # Fonts
        self.font_title = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_sub = pygame.font.SysFont("Arial", 11, bold=True)
        self.font_regular = pygame.font.SysFont("Arial", 12)
        self.font_bold = pygame.font.SysFont("Arial", 12, bold=True)
        self.font_telemetry = pygame.font.SysFont("Consolas", 12)

        # Color Palette
        self.COLOR_BG = (20, 22, 26)
        self.COLOR_PANEL_BG = (30, 33, 40)
        self.COLOR_BOX_BG = (22, 24, 28)
        self.COLOR_WORLD_BG = (42, 58, 38)   # Outdoor grass terrain
        self.COLOR_OBSTACLE = (50, 45, 40)
        self.COLOR_OBSTACLE_OUTLINE = (30, 25, 20)
        self.COLOR_PATH = (0, 230, 230)      # Cyan planned path
        self.COLOR_TRAIL = (255, 215, 0)     # Gold historical trail
        self.COLOR_ROBOT = (230, 90, 40)     # Orange chassis
        self.COLOR_GOAL = (50, 220, 90)      # Bright green goal
        self.COLOR_START = (60, 130, 240)    # Blue start
        self.COLOR_TEXT = (230, 235, 240)
        self.COLOR_TEXT_DIM = (150, 160, 170)
        self.COLOR_ACCENT = (0, 180, 255)

    def world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        """Converts continuous world meters (x, y) to Pygame screen pixel coordinates."""
        sx = int(self.offset_x + wx * self.scale)
        sy = int(self.offset_y + (self.world_height - wy) * self.scale)
        return (sx, sy)

    def screen_to_world(self, sx: int, sy: int) -> Optional[Tuple[float, float]]:
        """Converts Pygame click coordinates back to continuous world meters."""
        if not self.world_view_rect.collidepoint(sx, sy):
            return None
        wx = (sx - self.offset_x) / self.scale
        wy = self.world_height - (sy - self.offset_y) / self.scale
        wx = max(0.5, min(self.world_width - 0.5, wx))
        wy = max(0.5, min(self.world_height - 0.5, wy))
        return (wx, wy)

    def render(
        self,
        world: SimulationWorld,
        robot: RobotSim,
        camera_frame: CameraFrame,
        detection: ObstacleDetection,
        estimated_pose: Pose2D,
        occupancy_grid: GridMap,
        active_path: Path2D,
        twist_cmd: TwistCommand,
        status_text: str,
        fps: float,
        planned_path_length: float = 0.0,
        replan_count: int = 0,
        driven_length: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        """
        Renders complete multi-panel dashboard and handles mouse click events.
        """
        clicked_goal = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                clicked_goal = self.screen_to_world(mouse_pos[0], mouse_pos[1])

        self.screen.fill(self.COLOR_BG)

        # 1. Left Panel: Top-Down World Navigation Map
        self._render_world_panel(world, robot, estimated_pose, active_path)

        # 2. Right Panel Top: 4-Stage Perception Pipeline
        self._render_perception_pipeline_panel(camera_frame, detection)

        # 3. Right Panel Bottom Left: Perceived Occupancy Grid / Costmap
        self._render_costmap_panel(occupancy_grid, robot.pose)

        # 4. Right Panel Bottom Right: Telemetry & Controls
        self._render_telemetry_panel(
            robot.pose,
            estimated_pose,
            twist_cmd,
            status_text,
            fps,
            planned_path_length=planned_path_length,
            num_waypoints=len(active_path.waypoints),
            replan_count=replan_count,
            driven_length=driven_length,
        )

        pygame.display.flip()
        self.clock.tick(60)

        return clicked_goal

    def _render_world_panel(
        self,
        world: SimulationWorld,
        robot: RobotSim,
        estimated_pose: Pose2D,
        active_path: Path2D,
    ) -> None:
        """Draws top-down outdoor environment, obstacles, UGV chassis, and path."""
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BG, self.world_view_rect, border_radius=8)
        inner_rect = self.world_view_rect.inflate(-8, -8)
        pygame.draw.rect(self.screen, self.COLOR_WORLD_BG, inner_rect, border_radius=6)

        # Draw grid lines (1m intervals)
        for gx in range(0, int(self.world_width) + 1, 2):
            p1 = self.world_to_screen(gx, 0)
            p2 = self.world_to_screen(gx, self.world_height)
            pygame.draw.line(self.screen, (52, 72, 48), p1, p2, 1)

        for gy in range(0, int(self.world_height) + 1, 2):
            p1 = self.world_to_screen(0, gy)
            p2 = self.world_to_screen(self.world_width, gy)
            pygame.draw.line(self.screen, (52, 72, 48), p1, p2, 1)

        # Draw obstacles
        for obs in world.obstacles:
            sc = self.world_to_screen(obs.x, obs.y)
            sr = int(obs.radius * self.scale)
            # Drop shadow
            pygame.draw.circle(self.screen, (28, 38, 25), (sc[0] + 3, sc[1] + 3), sr)
            # Obstacle circle
            pygame.draw.circle(self.screen, obs.color_rgb, sc, sr)
            pygame.draw.circle(self.screen, (25, 20, 18), sc, sr, 2)

        # Historical Trajectory Trail
        if len(robot.trajectory_history) > 1:
            trail_pts = [self.world_to_screen(p[0], p[1]) for p in robot.trajectory_history]
            pygame.draw.lines(self.screen, self.COLOR_TRAIL, False, trail_pts, 2)

        # Planned A* Path
        if active_path.is_valid and not active_path.is_empty():
            path_pts = [self.world_to_screen(wp[0], wp[1]) for wp in active_path.waypoints]
            if len(path_pts) > 1:
                pygame.draw.lines(self.screen, self.COLOR_PATH, False, path_pts, 3)
            for pt in path_pts:
                pygame.draw.circle(self.screen, self.COLOR_PATH, pt, 3)

        # Goal & Start points
        g_scr = self.world_to_screen(world.goal_pos[0], world.goal_pos[1])
        pygame.draw.circle(self.screen, self.COLOR_GOAL, g_scr, 9)
        pygame.draw.circle(self.screen, (255, 255, 255), g_scr, 9, 2)
        txt_g = self.font_bold.render("GOAL", True, (255, 255, 255))
        self.screen.blit(txt_g, (g_scr[0] + 12, g_scr[1] - 8))

        s_scr = self.world_to_screen(world.start_pose[0], world.start_pose[1])
        pygame.draw.circle(self.screen, self.COLOR_START, s_scr, 7)
        pygame.draw.circle(self.screen, (255, 255, 255), s_scr, 7, 2)

        # FOV cone and chassis
        self._draw_fov_cone(robot.pose)
        self._draw_robot_chassis(robot.pose)

        title = self.font_title.render("TOP-DOWN WORLD & GLOBAL NAVIGATION MAP", True, self.COLOR_TEXT)
        self.screen.blit(title, (self.world_view_rect.x + 12, self.world_view_rect.y + 12))

    def _draw_fov_cone(self, pose: Pose2D) -> None:
        """Draws transparent camera FOV cone."""
        fov_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        cam_range = 6.5
        half_fov = math.radians(35.0)

        p_center = self.world_to_screen(pose.x, pose.y)
        p_left = self.world_to_screen(
            pose.x + cam_range * math.cos(pose.theta + half_fov),
            pose.y + cam_range * math.sin(pose.theta + half_fov),
        )
        p_right = self.world_to_screen(
            pose.x + cam_range * math.cos(pose.theta - half_fov),
            pose.y + cam_range * math.sin(pose.theta - half_fov),
        )

        pygame.draw.polygon(fov_surface, (255, 255, 180, 40), [p_center, p_left, p_right])
        pygame.draw.line(fov_surface, (255, 255, 180, 90), p_center, p_left, 1)
        pygame.draw.line(fov_surface, (255, 255, 180, 90), p_center, p_right, 1)
        self.screen.blit(fov_surface, (0, 0))

    def _draw_robot_chassis(self, pose: Pose2D) -> None:
        """Draws UGV chassis and direction pointer."""
        center = self.world_to_screen(pose.x, pose.y)
        l_px = int(0.7 * self.scale)
        w_px = int(0.5 * self.scale)

        corners = [
            (l_px / 2, -w_px / 2),
            (l_px / 2, w_px / 2),
            (-l_px / 2, w_px / 2),
            (-l_px / 2, -w_px / 2),
        ]

        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)

        rotated = []
        for lx, ly in corners:
            rx = center[0] + (lx * cos_t + ly * sin_t)
            ry = center[1] - (lx * sin_t - ly * cos_t)
            rotated.append((rx, ry))

        pygame.draw.polygon(self.screen, self.COLOR_ROBOT, rotated)
        pygame.draw.polygon(self.screen, (255, 255, 255), rotated, 2)

        nose_x = center[0] + int((l_px / 2 + 8) * cos_t)
        nose_y = center[1] - int((l_px / 2 + 8) * sin_t)
        pygame.draw.line(self.screen, (255, 255, 255), center, (nose_x, nose_y), 3)

    def _render_perception_pipeline_panel(self, frame: CameraFrame, detection: ObstacleDetection) -> None:
        """
        Renders the 4-Stage Perception Visualization Panel:
        1. ORIGINAL CAMERA IMAGE
        2. OBSTACLE MASK
        3. TRAVERSABLE TERRAIN MASK
        4. DETECTED OBSTACLE REGIONS (ANNOTATED)
        """
        panel_rect = pygame.Rect(self.right_panel_x, 10, 690, 395)
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BG, panel_rect, border_radius=8)

        title = self.font_title.render("4-STAGE PERCEPTION PIPELINE (CLASSICAL CV)", True, self.COLOR_TEXT)
        self.screen.blit(title, (panel_rect.x + 12, panel_rect.y + 8))

        sub_w, sub_h = 328, 160
        pos1 = (panel_rect.x + 12, panel_rect.y + 32)            # Top-Left: Original
        pos2 = (panel_rect.x + 350, panel_rect.y + 32)           # Top-Right: Obstacle Mask
        pos3 = (panel_rect.x + 12, panel_rect.y + 212)           # Bottom-Left: Traversable Mask
        pos4 = (panel_rect.x + 350, panel_rect.y + 212)          # Bottom-Right: Annotated Regions

        # 1. Original Camera Image
        if frame.image is not None:
            r1 = cv2.resize(frame.image, (sub_w, sub_h))
            s1 = pygame.surfarray.make_surface(r1.swapaxes(0, 1))
            self.screen.blit(s1, pos1)
        self._draw_subpanel_border(pos1, sub_w, sub_h, "1. ORIGINAL CAMERA FEED (RGB)")

        # 2. Obstacle Mask
        if detection.obstacle_mask is not None:
            obs_colored = cv2.cvtColor(detection.obstacle_mask, cv2.COLOR_GRAY2RGB)
            # Tint obstacles in red
            obs_colored[detection.obstacle_mask > 0] = [230, 40, 40]
            r2 = cv2.resize(obs_colored, (sub_w, sub_h))
            s2 = pygame.surfarray.make_surface(r2.swapaxes(0, 1))
            self.screen.blit(s2, pos2)
        self._draw_subpanel_border(pos2, sub_w, sub_h, "2. OBSTACLE MASK (SEGMENTED)")

        # 3. Traversable Terrain Mask
        if detection.traversable_mask is not None:
            trav_colored = cv2.cvtColor(detection.traversable_mask, cv2.COLOR_GRAY2RGB)
            # Tint traversable ground in emerald green
            trav_colored[detection.traversable_mask > 0] = [40, 200, 90]
            r3 = cv2.resize(trav_colored, (sub_w, sub_h))
            s3 = pygame.surfarray.make_surface(r3.swapaxes(0, 1))
            self.screen.blit(s3, pos3)
        self._draw_subpanel_border(pos3, sub_w, sub_h, "3. TRAVERSABLE TERRAIN MASK")

        # 4. Detected Obstacle Regions (Annotated)
        annotated_src = detection.annotated_image if detection.annotated_image is not None else frame.image
        if annotated_src is not None:
            r4 = cv2.resize(annotated_src, (sub_w, sub_h))
            s4 = pygame.surfarray.make_surface(r4.swapaxes(0, 1))
            self.screen.blit(s4, pos4)
        self._draw_subpanel_border(pos4, sub_w, sub_h, "4. DETECTED OBSTACLE REGIONS (HUD)")

    def _draw_subpanel_border(self, pos: Tuple[int, int], w: int, h: int, label: str) -> None:
        """Draws bounding box border and text title banner for sub-viewports."""
        rect = pygame.Rect(pos[0], pos[1], w, h)
        pygame.draw.rect(self.screen, (60, 65, 75), rect, 1)

        # Title background tag
        tag_surf = self.font_sub.render(f" {label} ", True, (255, 255, 255))
        tag_rect = tag_surf.get_rect(topleft=(pos[0] + 4, pos[1] + 4))
        bg_rect = tag_rect.inflate(4, 2)
        pygame.draw.rect(self.screen, (20, 25, 30, 220), bg_rect, border_radius=3)
        self.screen.blit(tag_surf, (pos[0] + 4, pos[1] + 4))

    def _render_costmap_panel(self, grid: GridMap, robot_pose: Pose2D) -> None:
        """Renders live perceived occupancy grid and inflation costmap."""
        panel_rect = pygame.Rect(self.right_panel_x, 415, 335, 395)
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BG, panel_rect, border_radius=8)

        title = self.font_title.render("PERCEIVED OCCUPANCY GRID", True, self.COLOR_TEXT)
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 8))

        if grid.data is not None:
            view_w, view_h = 315, 345
            cmap = cv2.applyColorMap(grid.data, cv2.COLORMAP_JET)
            cmap[grid.data == 0] = [35, 35, 45]       # Dark blue-gray for free space
            cmap[grid.data >= 65] = [0, 0, 220]      # Red for lethal obstacles

            resized_map = cv2.resize(cmap, (view_w, view_h), interpolation=cv2.INTER_NEAREST)
            resized_map = cv2.flip(resized_map, 0)   # Flip cartesian Y
            surf = pygame.surfarray.make_surface(resized_map.swapaxes(0, 1))
            self.screen.blit(surf, (panel_rect.x + 10, panel_rect.y + 35))

    def _render_telemetry_panel(
        self,
        true_pose: Pose2D,
        est_pose: Pose2D,
        cmd: TwistCommand,
        status: str,
        fps: float,
        planned_path_length: float = 0.0,
        num_waypoints: int = 0,
        replan_count: int = 0,
        driven_length: float = 0.0,
    ) -> None:
        """Renders telemetry, path metrics, and navigation controls subpanel."""
        panel_rect = pygame.Rect(self.right_panel_x + 345, 415, 345, 395)
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BG, panel_rect, border_radius=8)

        title = self.font_title.render("TELEMETRY & STATUS", True, self.COLOR_TEXT)
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 8))

        drift_dist = math.hypot(true_pose.x - est_pose.x, true_pose.y - est_pose.y)
        drift_theta = math.degrees(abs(true_pose.theta - est_pose.theta))

        telemetry_lines = [
            f"STATE       : {status.upper()}",
            f"FPS         : {fps:.1f} FPS",
            f"PLANNED LEN : {planned_path_length:5.2f} m ({num_waypoints} wps)",
            f"REPLANS     : {replan_count:3d} dynamic updates",
            f"DRIVEN DIST : {driven_length:5.2f} m",
            f"TRUE POSE   : X={true_pose.x:4.2f}m | Y={true_pose.y:4.2f}m",
            f"EST POSE    : X={est_pose.x:4.2f}m | Y={est_pose.y:4.2f}m",
            f"ODOM DRIFT  : {drift_dist:4.2f}m ({drift_theta:3.1f}°)",
            f"COMMAND     : v={cmd.linear_v:4.2f}m/s | ω={math.degrees(cmd.angular_w):+5.1f}°/s",
        ]

        y_offset = panel_rect.y + 32
        for line in telemetry_lines:
            if "STATE" in line:
                color = self.COLOR_ACCENT
            elif "PLANNED" in line or "REPLANS" in line:
                color = (0, 230, 230)  # Cyan for path metrics
            else:
                color = self.COLOR_TEXT
            surf = self.font_telemetry.render(line, True, color)
            self.screen.blit(surf, (panel_rect.x + 10, y_offset))
            y_offset += 20

        # Instructions / Help
        divider_y = panel_rect.y + 225
        pygame.draw.line(self.screen, (50, 55, 65), (panel_rect.x + 8, divider_y), (panel_rect.x + 337, divider_y), 1)

        help1 = self.font_regular.render("🖱️ Left Click anywhere to set Destination Goal", True, (255, 215, 0))
        help2 = self.font_regular.render("📷 Perception uses ONLY the camera frame", True, self.COLOR_TEXT_DIM)
        help3 = self.font_regular.render("🗺️ Map builds dynamically as UGV explores", True, self.COLOR_TEXT_DIM)
        help4 = self.font_regular.render("⚡ A* dynamically replans when obstacles appear", True, self.COLOR_TEXT_DIM)

        self.screen.blit(help1, (panel_rect.x + 10, divider_y + 8))
        self.screen.blit(help2, (panel_rect.x + 10, divider_y + 28))
        self.screen.blit(help3, (panel_rect.x + 10, divider_y + 48))
        self.screen.blit(help4, (panel_rect.x + 10, divider_y + 68))
