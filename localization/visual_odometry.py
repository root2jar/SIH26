"""
Monocular Visual Odometry (VO) Module
Estimates 2D motion (translation and yaw rotation) between consecutive camera frames
using Lucas-Kanade Optical Flow feature tracking (OpenCV).
"""

import math
from typing import Optional
import cv2
import numpy as np
from localization.base import BaseLocalizer
from utils.data_types import CameraFrame, Pose2D
from utils.math_helpers import normalize_angle


class VisualOdometry(BaseLocalizer):
    """
    Estimates the UGV position (x, y) and heading (theta) relative to the starting position
    by tracking visual features across consecutive camera frames without GPS.
    """

    def __init__(self, initial_pose: Optional[Pose2D] = None, max_features: int = 150):
        self.max_features = max_features
        self.pose = initial_pose.copy() if initial_pose else Pose2D()
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None

        # Feature detector parameters (Shi-Tomasi corner detector)
        self.feature_params = dict(
            maxCorners=self.max_features,
            qualityLevel=0.03,
            minDistance=10,
            blockSize=7,
        )

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def reset(self, initial_pose: Optional[Pose2D] = None) -> None:
        self.pose = initial_pose.copy() if initial_pose else Pose2D()
        self.prev_gray = None
        self.prev_pts = None

    def update(self, frame: CameraFrame, delta_time: float) -> Pose2D:
        """
        Calculates relative motion from previous frame and updates accumulated pose.
        """
        if frame.image is None or frame.image.size == 0:
            return self.pose

        gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY)

        # First frame initialization
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 15:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return self.pose

        # Calculate optical flow between previous and current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        if curr_pts is not None and status is not None:
            # Select good points that were tracked in both frames
            good_prev = self.prev_pts[status == 1]
            good_curr = curr_pts[status == 1]

            if len(good_prev) >= 8:
                # Displacement vectors for tracked points: (dx, dy)
                displacements = good_curr - good_prev
                dx_mean = np.mean(displacements[:, 0])
                dy_mean = np.mean(displacements[:, 1])

                # Optical flow heuristics:
                # Horizontal shift (dx) corresponds to yaw angular change (dtheta)
                # Vertical downward flow (dy) corresponds to forward motion (ds)
                # Scaling factor maps image pixel flow to physical motion meters/radians
                img_width = frame.image.shape[1]
                fov_rad = math.radians(frame.fov_deg)

                # Angular rotation estimate (yaw)
                delta_theta = -(dx_mean / float(img_width)) * fov_rad * 0.85

                # Linear forward displacement estimate
                # Clamped to prevent erratic jumps from visual flicker
                delta_s = float(np.clip(dy_mean * 0.015, -0.2, 0.4))

                # Integrate kinematic motion
                avg_theta = self.pose.theta + delta_theta / 2.0
                self.pose.x += delta_s * math.cos(avg_theta)
                self.pose.y += delta_s * math.sin(avg_theta)
                self.pose.theta = normalize_angle(self.pose.theta + delta_theta)

        # Prepare for next frame
        # Re-detect new features if count drops
        new_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        self.prev_pts = new_pts if new_pts is not None else curr_pts
        self.prev_gray = gray

        return self.pose

    def get_pose(self) -> Pose2D:
        return self.pose
