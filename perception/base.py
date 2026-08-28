"""
Base Perception Interface
Defines the standard abstract contract for any vision-based obstacle/terrain perception system.
Allows seamless swapping between simple color segmentation, ML/PyTorch models (e.g. SegNet/YOLO),
or hardware stereo/depth cameras.
"""

from abc import ABC, abstractmethod
from utils.data_types import CameraFrame, ObstacleDetection


class BasePerception(ABC):
    """Abstract interface for all visual perception modules."""

    @abstractmethod
    def process_frame(self, frame: CameraFrame) -> ObstacleDetection:
        """
        Processes an incoming CameraFrame and returns detected obstacle coordinates
        in the UGV local body frame (meters relative to robot center).

        :param frame: CameraFrame containing optical image and metadata.
        :return: ObstacleDetection containing list of local (x_forward, y_left) obstacle coordinates.
        """
        pass
