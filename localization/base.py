"""
Base Localization Interface
Defines the standard abstract contract for visual odometry and SLAM estimators.
Allows visual odometry algorithms to be swapped or fused with IMU/wheel encoders.
"""

from abc import ABC, abstractmethod
from typing import Optional
from utils.data_types import CameraFrame, Pose2D


class BaseLocalizer(ABC):
    """Abstract interface for GPS-denied state estimation."""

    @abstractmethod
    def reset(self, initial_pose: Optional[Pose2D] = None) -> None:
        """Resets the pose estimate to start position."""
        pass

    @abstractmethod
    def update(self, frame: CameraFrame, delta_time: float) -> Pose2D:
        """
        Updates the pose estimate based on the newly received CameraFrame.
        :param frame: Current CameraFrame.
        :param delta_time: Elapsed time since last frame in seconds.
        :return: Estimated Pose2D in world frame.
        """
        pass

    @abstractmethod
    def get_pose(self) -> Pose2D:
        """Returns the current estimated Pose2D."""
        pass
