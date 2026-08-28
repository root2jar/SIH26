"""
Default Configuration for UGV Autonomous Navigation System
Central repository for simulation, sensor, mapping, planning, and control parameters.
"""

from dataclasses import dataclass, field


@dataclass
class WorldConfig:
    width: float = 24.0          # World width in meters
    height: float = 18.0         # World height in meters
    num_obstacles: int = 15      # Number of random/preset rocks & trees
    boundary_thickness: float = 0.5


@dataclass
class RobotConfig:
    wheelbase: float = 0.45      # Distance between wheels (m)
    radius: float = 0.35         # Bounding radius of UGV chassis (m)
    max_linear_speed: float = 1.2  # Max forward speed (m/s)
    max_angular_speed: float = 1.8 # Max turn rate (rad/s)
    accel_linear: float = 1.5     # Linear acceleration limit (m/s^2)
    accel_angular: float = 3.0    # Angular acceleration limit (rad/s^2)


@dataclass
class CameraConfig:
    image_width: int = 320       # Simulated/real camera resolution width
    image_height: int = 240      # Simulated/real camera resolution height
    fov_deg: float = 70.0        # Horizontal Field of View (degrees)
    max_depth_range: float = 8.0 # Maximum effective optical range (meters)
    camera_height: float = 0.45  # Height above ground (meters)
    camera_pitch_deg: float = 0.0 # Tilt angle downwards


@dataclass
class PerceptionConfig:
    # Color thresholding ranges in HSV for outdoor terrain vs obstacle detection
    # Drivable terrain (e.g., dirt / sandy path)
    drivable_hsv_low: tuple = (10, 30, 60)
    drivable_hsv_high: tuple = (35, 255, 220)
    # Obstacle color ranges (rocks, dark bushes, barriers)
    obstacle_hsv_low: tuple = (0, 0, 0)
    obstacle_hsv_high: tuple = (180, 255, 80)
    min_contour_area: int = 250


@dataclass
class MappingConfig:
    resolution: float = 0.10     # Grid resolution: 0.10m (10cm) per grid cell for fine corridor fidelity
    inflation_radius: float = 0.48 # Safety margin: robot radius (0.35m) + clearance buffer (0.13m)
    occupied_threshold: int = 65 # Cost >= 65 considered obstacle
    free_threshold: int = 25     # Cost <= 25 considered free space


@dataclass
class PlannerConfig:
    heuristic_weight: float = 1.0 # Optimal admissible Euclidean heuristic in A*
    safety_penalty_weight: float = 0.5 # Gentle clearance preference without excessive detours
    goal_tolerance: float = 0.4   # Goal arrival threshold (meters)
    replan_distance_threshold: float = 3.5 # Forward lookahead horizon to check for path blockage (meters)
    allow_diagonal: bool = True


@dataclass
class ControllerConfig:
    lookahead_distance: float = 0.75 # Pure pursuit lookahead distance (m)
    waypoint_tolerance: float = 0.55 # Radius to consider waypoint reached and advance to next (m)
    kp_angular: float = 2.2         # Proportional gain for steering
    target_speed: float = 0.9       # Desired cruising velocity (m/s)


@dataclass
class GUIConfig:
    screen_width: int = 1400
    screen_height: int = 820
    fps: int = 60
    world_scale: float = 38.0       # Pixels per meter in world view panel


@dataclass
class UGVSystemConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)


# Global default instance
DEFAULT_CONFIG = UGVSystemConfig()
