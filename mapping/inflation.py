"""
Obstacle Inflation Layer
Applies safety buffer margins around detected obstacles so that path planners
produce paths that maintain a safe physical clearance for the UGV chassis.
"""

import math
import cv2
import numpy as np


class ObstacleInflationLayer:
    """
    Inflates obstacle cells using a circular safety radius.
    Produces a graded costmap where:
    - 100: Lethal obstacle center
    - 50-99: Safety buffer / inflation zone (penalized by path planner)
    - 0: Completely free space
    """

    def __init__(self, resolution: float = 0.15, inflation_radius: float = 0.5):
        self.resolution = resolution
        self.inflation_radius = inflation_radius
        self.radius_cells = max(1, int(math.ceil(inflation_radius / resolution)))

        # Create circular structuring element for morphological dilation
        k_size = 2 * self.radius_cells + 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    def inflate(self, raw_grid: np.ndarray) -> np.ndarray:
        """
        Takes raw 2D grid (0: free, 100: obstacle) and computes the inflated costmap.
        :param raw_grid: 2D numpy array of shape (H, W).
        :return: 2D numpy array with inflated obstacles and safety cost gradient.
        """
        h, w = raw_grid.shape
        # Binary mask of lethal obstacles (cost >= 65)
        lethal_mask = (raw_grid >= 65).astype(np.uint8)

        if not np.any(lethal_mask):
            return raw_grid.copy()

        # Dilate obstacles to encompass the robot's physical radius + safety buffer
        inflated_mask = cv2.dilate(lethal_mask, self.kernel)

        # Distance transform to create smooth safety margin gradient
        # cv2.distanceTransform computes distance of zero pixels to nearest zero pixel
        inv_mask = (1 - lethal_mask).astype(np.uint8)
        dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5) * self.resolution

        # Output costmap: 100 for lethal obstacle, decaying cost in buffer zone, 0 outside
        costmap = np.zeros((h, w), dtype=np.uint8)
        costmap[lethal_mask == 1] = 100

        buffer_zone = (inflated_mask == 1) & (lethal_mask == 0)
        # Linear cost decay from 90 at obstacle boundary down to 30 at outer inflation limit
        if np.any(buffer_zone):
            decay = 1.0 - (dist[buffer_zone] / self.inflation_radius)
            decay = np.clip(decay, 0.0, 1.0)
            costmap[buffer_zone] = (30 + decay * 60).astype(np.uint8)

        return costmap
