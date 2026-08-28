"""
Vision-Based Obstacle Detector
Converts 2D image detections into 2D ground-plane coordinates (x_forward, y_left)
in the UGV local coordinate frame.
"""

import math
from typing import List, Tuple
import cv2
import numpy as np
from perception.base import BasePerception
from perception.color_segmentation import ColorTerrainSegmenter
from utils.data_types import CameraFrame, ObstacleDetection, ObstacleRegion


class VisualObstacleDetector(BasePerception):
    """
    Pure vision-based obstacle detector using classical OpenCV segmentation and
    monocular ground-plane projective geometry. Does not access ground-truth simulator state.
    """

    def __init__(
        self,
        segmenter: ColorTerrainSegmenter = None,
        camera_height: float = 0.45,
        max_detection_range: float = 7.5,
    ):
        self.segmenter = segmenter or ColorTerrainSegmenter(min_area=60)
        self.camera_height = camera_height
        self.max_detection_range = max_detection_range

    def process_frame(self, frame: CameraFrame) -> ObstacleDetection:
        """
        Processes camera RGB frame and extracts obstacle masks, traversable terrain masks,
        annotated HUD image, and local metric obstacle coordinates.
        """
        if frame.image is None or frame.image.size == 0:
            return ObstacleDetection(timestamp=frame.timestamp)

        h, w, _ = frame.image.shape
        horizon_y = h // 2

        # 1. Multi-channel classical CV segmentation
        obs_mask, trav_mask = self.segmenter.segment(frame.image)

        # 2. Extract obstacle region contours
        contours, _ = cv2.findContours(obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        local_points: List[Tuple[float, float]] = []
        obstacle_regions: List[ObstacleRegion] = []

        half_fov_rad = math.radians(frame.fov_deg / 2.0)
        fx = (w / 2.0) / math.tan(half_fov_rad)

        # Prepare annotated image with telemetry & bounding boxes
        annotated = frame.image.copy()

        # Draw horizon guide line on annotated view
        cv2.line(annotated, (0, horizon_y), (w, horizon_y), (180, 200, 220), 1, cv2.LINE_AA)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.segmenter.min_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            center_u = bx + bw / 2.0
            bottom_v = by + bh

            # Monocular ground-plane pinhole distance projection
            delta_v = max(1.0, float(bottom_v - horizon_y))
            distance = (self.camera_height * (h - horizon_y)) / delta_v

            if distance > self.max_detection_range or distance < 0.25:
                continue

            # Relative horizontal bearing angle (+ left, - right)
            angle_rad = -((center_u - w / 2.0) / (w / 2.0)) * half_fov_rad

            # Physical radius estimate from apparent pixel width
            est_radius = (bw * distance) / (2.0 * fx)
            est_radius = float(np.clip(est_radius, 0.4, 2.0))

            # Estimated obstacle center in UGV body frame (+x forward, +y left)
            center_dist = distance + est_radius * 0.75
            x_center = center_dist * math.cos(angle_rad)
            y_center = center_dist * math.sin(angle_rad)

            # Confidence score based on area and distance
            confidence = float(np.clip(1.0 - (distance / self.max_detection_range) * 0.4, 0.5, 0.98))

            region = ObstacleRegion(
                bbox=(bx, by, bw, bh),
                center_u=center_u,
                bottom_v=bottom_v,
                distance_meters=distance,
                local_x=x_center,
                local_y=y_center,
                radius_meters=est_radius,
                confidence=confidence,
                label="Obstacle",
            )
            obstacle_regions.append(region)
            local_points.append((x_center, y_center))

            # Draw HUD bounding box & label on annotated image
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            # Base ground contact point
            cv2.circle(annotated, (int(center_u), int(bottom_v)), 3, (0, 255, 255), -1)

            # Overlay telemetry tag
            tag = f"Obs: {distance:.1f}m | r={est_radius:.1f}m"
            cv2.putText(
                annotated,
                tag,
                (max(2, bx), max(12, by - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return ObstacleDetection(
            local_points=local_points,
            obstacle_regions=obstacle_regions,
            obstacle_mask=obs_mask,
            traversable_mask=trav_mask,
            annotated_image=annotated,
            confidence=1.0 if obstacle_regions else 0.0,
            timestamp=frame.timestamp,
        )
