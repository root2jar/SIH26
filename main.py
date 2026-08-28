"""
Main Pipeline Orchestration Loop for UGV Autonomous Navigation
Smart India Hackathon (SIH 2026) Prototype

Integrates modular pipelines:
1. Synthetic / Hardware Camera
2. Visual Obstacle & Terrain Perception
3. GPS-Denied Visual Odometry Localization
4. 2D Occupancy Grid Mapping & Safety Inflation
5. A* Path Planning & Dynamic Replanning
6. Pure Pursuit Motion Control
7. Real-Time Pygame Multi-Panel GUI
"""

import time
import math
from configs.default_config import DEFAULT_CONFIG
from utils.data_types import Pose2D, TwistCommand
from utils.math_helpers import euclidean_distance
from simulation.world import SimulationWorld
from simulation.robot_sim import RobotSim
from simulation.synthetic_camera import SyntheticCamera
from simulation.simulator_gui import SimulatorGUI
from perception.obstacle_detector import VisualObstacleDetector
from localization.visual_odometry import VisualOdometry
from mapping.occupancy_grid import OccupancyGridMap
from planning.a_star import AStarPlanner
from planning.dynamic_replanner import DynamicReplanner
from control.pure_pursuit import PurePursuitController


def main():
    print("=" * 70)
    print("  SIH 2026: Vision-Based Autonomous Navigation for UGV (GPS-Denied)")
    print("=" * 70)
    print("Initializing modular subsystems...\n")

    cfg = DEFAULT_CONFIG

    # 1. Initialize Simulation Environment
    world = SimulationWorld(width=cfg.world.width, height=cfg.world.height)
    start_pose = Pose2D(
        x=world.start_pose[0],
        y=world.start_pose[1],
        theta=world.start_pose[2],
    )
    robot = RobotSim(
        initial_pose=start_pose,
        wheelbase=cfg.robot.wheelbase,
        max_linear_speed=cfg.robot.max_linear_speed,
        max_angular_speed=cfg.robot.max_angular_speed,
    )
    camera = SyntheticCamera(
        width=cfg.camera.image_width,
        height=cfg.camera.image_height,
        fov_deg=cfg.camera.fov_deg,
        max_range=cfg.camera.max_depth_range,
        camera_height=cfg.camera.camera_height,
    )

    # 2. Initialize Autonomy Stack Modules
    perception = VisualObstacleDetector(
        camera_height=cfg.camera.camera_height,
        max_detection_range=cfg.camera.max_depth_range,
    )
    localizer = VisualOdometry(initial_pose=start_pose)
    occupancy_map = OccupancyGridMap(
        width_meters=cfg.world.width,
        height_meters=cfg.world.height,
        resolution=cfg.mapping.resolution,
        inflation_radius=cfg.mapping.inflation_radius,
    )
    # Seed boundary walls only (Environment obstacles are discovered purely via vision)
    occupancy_map.add_boundary_walls(thickness_meters=0.4)

    planner = AStarPlanner(
        heuristic_weight=cfg.planner.heuristic_weight,
        allow_diagonal=cfg.planner.allow_diagonal,
        occupied_threshold=cfg.mapping.occupied_threshold,
    )
    replanner = DynamicReplanner(
        planner=planner,
        collision_check_horizon=cfg.planner.replan_distance_threshold,
        occupied_threshold=cfg.mapping.occupied_threshold,
    )
    controller = PurePursuitController(
        lookahead_distance=cfg.controller.lookahead_distance,
        target_speed=cfg.controller.target_speed,
        max_angular_speed=cfg.robot.max_angular_speed,
        waypoint_tolerance=cfg.controller.waypoint_tolerance,
        goal_tolerance=cfg.planner.goal_tolerance,
        kp_angular=cfg.controller.kp_angular,
    )

    # 3. Initialize Interactive GUI Dashboard
    gui = SimulatorGUI(
        screen_width=cfg.gui.screen_width,
        screen_height=cfg.gui.screen_height,
        world_width=cfg.world.width,
        world_height=cfg.world.height,
    )

    # Initial plan towards goal
    initial_inflated = occupancy_map.get_inflated_grid()
    active_path = replanner.set_goal(start_pose, world.goal_pos, initial_inflated)

    print("Subsystems initialized successfully!")
    print(f"Goal set at: X={world.goal_pos[0]:.1f}m, Y={world.goal_pos[1]:.1f}m")
    print("Starting navigation loop...\n")

    last_time = time.time()
    frame_count = 0
    fps = 60.0
    status_text = "Navigating"

    # Path & navigation performance metrics
    replan_count = 0
    driven_length = 0.0
    last_robot_pos = robot.pose.position_tuple()

    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            if dt <= 0:
                dt = 1.0 / 60.0
            last_time = current_time

            # Update FPS calculation
            frame_count += 1
            if frame_count % 15 == 0:
                fps = 1.0 / dt if dt > 0 else 60.0

            # -------------------------------------------------------------
            # STEP 1: SENSING (Simulated or Real Hardware Optical Camera)
            # -------------------------------------------------------------
            camera_frame = camera.render(robot.pose, world, timestamp=current_time)

            # -------------------------------------------------------------
            # STEP 2: PERCEPTION (Vision-Only Terrain & Obstacle Segmentation)
            # -------------------------------------------------------------
            detection = perception.process_frame(camera_frame)

            # -------------------------------------------------------------
            # STEP 3: LOCALIZATION (GPS-Denied Visual Odometry)
            # -------------------------------------------------------------
            estimated_pose = localizer.update(camera_frame, dt)

            # -------------------------------------------------------------
            # STEP 4: MAPPING (Vision-Derived Occupancy Grid & Inflation)
            # -------------------------------------------------------------
            occupancy_map.update(robot.pose, detection)
            inflated_grid = occupancy_map.get_inflated_grid()

            # -------------------------------------------------------------
            # STEP 5: PLANNING & DYNAMIC REPLANNING (Avoid Discovered Obstacles)
            # -------------------------------------------------------------
            active_path, replanned = replanner.update(robot.pose, inflated_grid, timestamp=current_time)
            if replanned:
                replan_count += 1
                status_text = "Dynamic Replanning"

            # Compute planned path length
            if len(active_path.waypoints) > 1:
                planned_path_length = sum(
                    euclidean_distance(active_path.waypoints[i], active_path.waypoints[i + 1])
                    for i in range(len(active_path.waypoints) - 1)
                )
            else:
                planned_path_length = 0.0

            # -------------------------------------------------------------
            # STEP 6: MOTION CONTROL (Pure Pursuit Path Follower)
            # -------------------------------------------------------------
            twist_cmd, goal_reached = controller.compute_command(robot.pose, active_path, dt)

            if goal_reached:
                status_text = "Goal Reached!"
                twist_cmd = TwistCommand(0.0, 0.0)
            elif not replanned:
                status_text = "Navigating"

            # -------------------------------------------------------------
            # STEP 7: ACTUATION (Physics Integration & Physical Collision Guard)
            # -------------------------------------------------------------
            robot.step(twist_cmd, dt, world=world, robot_radius=cfg.robot.radius)

            # Accumulate physical driven distance
            cur_pos = robot.pose.position_tuple()
            driven_length += euclidean_distance(last_robot_pos, cur_pos)
            last_robot_pos = cur_pos

            # -------------------------------------------------------------
            # STEP 8: GUI DASHBOARD & USER INTERACTION
            # -------------------------------------------------------------
            clicked_goal = gui.render(
                world=world,
                robot=robot,
                camera_frame=camera_frame,
                detection=detection,
                estimated_pose=estimated_pose,
                occupancy_grid=inflated_grid,
                active_path=active_path,
                twist_cmd=twist_cmd,
                status_text=status_text,
                fps=fps,
                planned_path_length=planned_path_length,
                replan_count=replan_count,
                driven_length=driven_length,
            )

            # Interactive goal setting via mouse click
            if clicked_goal is not None:
                world.goal_pos = clicked_goal
                active_path = replanner.set_goal(robot.pose, clicked_goal, inflated_grid)
                controller.reset_path_progress()
                status_text = "New Goal Set"
                print(f"[GUI] New Destination Goal Selected: X={clicked_goal[0]:.2f}m, Y={clicked_goal[1]:.2f}m")

    except KeyboardInterrupt:
        print("\nNavigation pipeline terminated by user.")


if __name__ == "__main__":
    main()
