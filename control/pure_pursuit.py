"""
Pure Pursuit Path Tracking Controller
Standard geometric trajectory tracking algorithm for differential drive mobile robots.
Calculates steering angle and forward throttle based on a lookahead target point.
"""

import math
from typing import Tuple, Optional
from control.base import BaseController
from utils.data_types import Pose2D, Path2D, TwistCommand
from utils.math_helpers import world_to_robot, euclidean_distance, normalize_angle


class PurePursuitController(BaseController):
    """
    Computes (v, omega) commands to steer the UGV smoothly towards waypoints on the path.
    Uses forward waypoint progress indexing, waypoint tolerance radius, and heading alignment.
    """

    def __init__(
        self,
        lookahead_distance: float = 0.8,
        target_speed: float = 0.9,
        max_angular_speed: float = 1.8,
        waypoint_tolerance: float = 0.6,
        goal_tolerance: float = 0.35,
        kp_angular: float = 2.2,
    ):
        self.lookahead_distance = lookahead_distance
        self.target_speed = target_speed
        self.max_angular_speed = max_angular_speed
        self.waypoint_tolerance = waypoint_tolerance
        self.goal_tolerance = goal_tolerance
        self.kp_angular = kp_angular

        self.current_wp_idx: int = 0
        self._last_path_len: int = 0
        self._last_start_wp: Optional[Tuple[float, float]] = None

    def reset_path_progress(self) -> None:
        """Resets the waypoint tracking index."""
        self.current_wp_idx = 0
        self._last_path_len = 0
        self._last_start_wp = None

    def compute_command(
        self,
        current_pose: Pose2D,
        path: Path2D,
        delta_time: float,
    ) -> Tuple[TwistCommand, bool]:
        """
        Calculates linear and angular velocity commands to follow waypoints cleanly.
        """
        if path.is_empty() or not path.is_valid:
            return TwistCommand(0.0, 0.0), False

        num_wp = len(path.waypoints)
        final_goal = path.waypoints[-1]
        cur_pos = current_pose.position_tuple()

        # Reset waypoint progress index if a new/replanned trajectory was provided
        if (
            num_wp != self._last_path_len
            or self._last_start_wp != path.waypoints[0]
            or self.current_wp_idx >= num_wp
        ):
            self.current_wp_idx = 0
            self._last_path_len = num_wp
            self._last_start_wp = path.waypoints[0]

        # Check if UGV reached the final destination goal
        dist_to_final = euclidean_distance(cur_pos, final_goal)
        if dist_to_final <= self.goal_tolerance:
            return TwistCommand(0.0, 0.0), True

        # Advance current waypoint if UGV is within waypoint tolerance radius
        while self.current_wp_idx < num_wp - 1:
            wp = path.waypoints[self.current_wp_idx]
            dist_to_wp = euclidean_distance(cur_pos, wp)
            if dist_to_wp < self.waypoint_tolerance:
                self.current_wp_idx += 1
            else:
                break

        # Find target lookahead point (strictly forward from current active waypoint index)
        target_point = path.waypoints[self.current_wp_idx]
        for i in range(self.current_wp_idx, num_wp):
            wp = path.waypoints[i]
            d = euclidean_distance(cur_pos, wp)
            if d >= self.lookahead_distance:
                target_point = wp
                break
            target_point = wp

        # Transform target point to robot's local body frame (+x forward, +y left)
        local_x, local_y = world_to_robot(target_point, current_pose)

        # Heading error angle to target in robot frame [-pi, pi]
        alpha = math.atan2(local_y, local_x)

        # Proportional steering angular velocity
        angular_w = self.kp_angular * alpha
        angular_w = max(-self.max_angular_speed, min(self.max_angular_speed, angular_w))

        # Adaptive linear velocity control:
        # If heading error is large (> 65 deg), turn in place before accelerating
        if abs(alpha) > math.radians(65.0):
            linear_v = 0.0
        else:
            # Smooth throttle scaling based on heading alignment and approach distance
            turn_factor = math.cos(alpha) ** 2
            approach_factor = min(1.0, dist_to_final / 1.5)
            linear_v = self.target_speed * turn_factor * approach_factor
            # Maintain minimum forward momentum if not yet at destination
            if dist_to_final > self.goal_tolerance:
                linear_v = max(0.2, linear_v)

        return TwistCommand(linear_v=float(linear_v), angular_w=float(angular_w)), False
