"""
Educational Landmark-Based Visual SLAM (Simultaneous Localization and Mapping)
Builds upon Visual Odometry by registering observed obstacle landmarks in a 2D map
to correct for visual odometry drift over time.
"""

from typing import List, Tuple, Optional
import math
from utils.data_types import Pose2D, CameraFrame, ObstacleDetection
from utils.math_helpers import robot_to_world, euclidean_distance, normalize_angle
from localization.base import BaseLocalizer


class SimpleLandmarkSLAM(BaseLocalizer):
    """
    Educational 2D SLAM system:
    Tracks visual landmarks (obstacle centroid positions) and matches them to a persistent map,
    providing closed-loop pose corrections against odometry drift.
    """

    def __init__(self, initial_pose: Optional[Pose2D] = None, match_threshold: float = 0.8):
        self.pose = initial_pose.copy() if initial_pose else Pose2D()
        self.landmarks: List[Tuple[float, float]] = []  # Known landmark positions in world frame
        self.match_threshold = match_threshold

    def reset(self, initial_pose: Optional[Pose2D] = None) -> None:
        self.pose = initial_pose.copy() if initial_pose else Pose2D()
        self.landmarks.clear()

    def update(self, frame: CameraFrame, delta_time: float) -> Pose2D:
        """Fallback update if no landmarks provided."""
        return self.pose

    def update_with_detection(self, detection: ObstacleDetection, odometry_pose: Pose2D) -> Pose2D:
        """
        Fuses landmark observations from perception with odometry pose to reduce drift.
        """
        self.pose = odometry_pose.copy()

        for local_pt in detection.local_points:
            # Transform detected point to global world coordinates
            world_pt = robot_to_world(local_pt, self.pose)

            # Check if landmark matches an existing one in the map
            matched_idx = -1
            min_dist = float("inf")
            for idx, lm in enumerate(self.landmarks):
                d = euclidean_distance(world_pt, lm)
                if d < min_dist:
                    min_dist = d
                    matched_idx = idx

            if matched_idx >= 0 and min_dist < self.match_threshold:
                # Landmark matched! Use the correction error to adjust robot position
                lm_world = self.landmarks[matched_idx]
                correction_x = (lm_world[0] - world_pt[0]) * 0.15
                correction_y = (lm_world[1] - world_pt[1]) * 0.15
                self.pose.x += correction_x
                self.pose.y += correction_y
            else:
                # New landmark observed: insert into SLAM map
                self.landmarks.append(world_pt)

        return self.pose

    def get_pose(self) -> Pose2D:
        return self.pose
