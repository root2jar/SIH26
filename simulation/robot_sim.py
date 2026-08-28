"""
Differential Drive Robot Kinematic Simulator
Simulates the physical motion of the UGV using unicycle kinematics and acceleration limits.
"""

import math
from typing import List, Tuple
from utils.data_types import Pose2D, TwistCommand
from utils.math_helpers import normalize_angle


class RobotSim:
    """
    Simulates physical differential drive ground vehicle dynamics.
    """

    def __init__(
        self,
        initial_pose: Pose2D,
        wheelbase: float = 0.45,
        max_linear_speed: float = 1.2,
        max_angular_speed: float = 1.8,
        accel_linear: float = 2.0,
        accel_angular: float = 4.0,
    ):
        self.pose = initial_pose.copy()
        self.wheelbase = wheelbase
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.accel_linear = accel_linear
        self.accel_angular = accel_angular

        self.current_v: float = 0.0
        self.current_w: float = 0.0

        self.trajectory_history: List[Tuple[float, float]] = [self.pose.position_tuple()]

    def reset(self, pose: Pose2D) -> None:
        self.pose = pose.copy()
        self.current_v = 0.0
        self.current_w = 0.0
        self.trajectory_history = [self.pose.position_tuple()]

    def step(self, cmd: TwistCommand, dt: float, world=None, robot_radius: float = 0.35) -> Pose2D:
        """
        Integrates kinematics forward by dt seconds given the desired velocity command.
        Enforces physical collision boundaries against circular obstacles in the simulation world.
        """
        # Clamp command inputs to physical limits
        target_v = max(-self.max_linear_speed, min(self.max_linear_speed, cmd.linear_v))
        target_w = max(-self.max_angular_speed, min(self.max_angular_speed, cmd.angular_w))

        # Linear acceleration smoothing
        dv = target_v - self.current_v
        max_dv = self.accel_linear * dt
        self.current_v += max(-max_dv, min(max_dv, dv))

        # Angular acceleration smoothing
        dw = target_w - self.current_w
        max_dw = self.accel_angular * dt
        self.current_w += max(-max_dw, min(max_dw, dw))

        # Candidate kinematic integration
        cand_x = self.pose.x + self.current_v * math.cos(self.pose.theta) * dt
        cand_y = self.pose.y + self.current_v * math.sin(self.pose.theta) * dt
        cand_theta = normalize_angle(self.pose.theta + self.current_w * dt)

        # Physical simulation collision enforcement against circular obstacles
        if world is not None and hasattr(world, "obstacles"):
            for obs in world.obstacles:
                dx = cand_x - obs.x
                dy = cand_y - obs.y
                dist = math.hypot(dx, dy)
                min_allowed_dist = obs.radius + robot_radius
                if dist < min_allowed_dist:
                    # Enforce solid boundary: push out along radial normal vector
                    nx = dx / max(0.001, dist)
                    ny = dy / max(0.001, dist)
                    cand_x = obs.x + nx * min_allowed_dist
                    cand_y = obs.y + ny * min_allowed_dist
                    # Dissipate forward velocity upon contact
                    self.current_v = max(0.0, self.current_v * 0.1)

            # Enforce world perimeter boundaries
            cand_x = max(robot_radius + 0.4, min(world.width - robot_radius - 0.4, cand_x))
            cand_y = max(robot_radius + 0.4, min(world.height - robot_radius - 0.4, cand_y))

        self.pose.x = cand_x
        self.pose.y = cand_y
        self.pose.theta = cand_theta

        # Record trajectory path trail
        if len(self.trajectory_history) == 0 or math.hypot(
            self.pose.x - self.trajectory_history[-1][0],
            self.pose.y - self.trajectory_history[-1][1],
        ) > 0.05:
            self.trajectory_history.append(self.pose.position_tuple())

        return self.pose.copy()
