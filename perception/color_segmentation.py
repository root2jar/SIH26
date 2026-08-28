"""
Color & Texture-Based Terrain Segmentation
Educational baseline using OpenCV HSV thresholding and contour analysis
to separate drivable terrain from non-traversable obstacles (rocks, bushes, barriers).
"""

import cv2
import numpy as np
from typing import Tuple


class ColorTerrainSegmenter:
    """
    Performs multi-channel color filtering and morphological operations on camera frames
    to extract both an obstacle binary mask and a traversable terrain mask without assuming
    a single fixed obstacle color.
    """

    def __init__(self, min_area: int = 80):
        self.min_area = min_area

        # Morphological kernels for noise suppression and contour consolidation
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def segment(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segments the image into an obstacle binary mask and a traversable terrain mask.
        :param image_rgb: RGB image array (H, W, 3).
        :return: (obstacle_mask, traversable_mask) binary masks of shape (H, W).
        """
        h, w, _ = image_rgb.shape
        horizon_y = h // 2

        # Ground plane region of interest (below horizon)
        ground_roi = image_rgb[horizon_y:, :]

        # 1. Gaussian Blur to filter camera noise & high-frequency texture specks
        blurred = cv2.GaussianBlur(ground_roi, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

        # 2. Segment Traversable Ground (Green grass + sandy dirt trails)
        # Grass range: Hue in [28, 90], Saturation >= 30, Value >= 50
        # Dirt/sand range: Hue in [10, 28], Saturation >= 20, Value >= 60
        grass_mask = cv2.inRange(hsv, (28, 30, 50), (90, 255, 255))
        dirt_mask = cv2.inRange(hsv, (10, 20, 60), (28, 255, 255))
        traversable_ground = cv2.bitwise_or(grass_mask, dirt_mask)

        # 3. Obstacle Mask: Foreground anomalies on ground (rocks, timber, dense shrubs, boulders)
        raw_obstacle = cv2.bitwise_not(traversable_ground)

        # 4. Morphological Filtering
        # Open: removes isolated single-pixel false alarms in grass
        obstacle_clean = cv2.morphologyEx(raw_obstacle, cv2.MORPH_OPEN, self.kernel_open)
        # Close: bridges gaps in obstacle contours to form solid connected objects
        obstacle_clean = cv2.morphologyEx(obstacle_clean, cv2.MORPH_CLOSE, self.kernel_close)

        # Re-derive clean traversable ground
        traversable_clean = cv2.bitwise_not(obstacle_clean)

        # Assemble full-sized (H, W) frame masks
        full_obstacle_mask = np.zeros((h, w), dtype=np.uint8)
        full_traversable_mask = np.zeros((h, w), dtype=np.uint8)

        full_obstacle_mask[horizon_y:, :] = obstacle_clean
        full_traversable_mask[horizon_y:, :] = traversable_clean

        return full_obstacle_mask, full_traversable_mask
