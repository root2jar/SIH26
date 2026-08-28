"""
Base Motion Controller Interface
Defines the standard abstract contract for trajectory tracking and velocity command generation.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from utils.data_types import Pose2D, Path2D, TwistCommand


class BaseController(ABC):
    """Abstract interface for path tracking controllers (Pure Pursuit, Stanley, MPC, PID)."""

    @abstractmethod
    def compute_command(
        self,
        current_pose: Pose2D,
        path: Path2D,
        delta_time: float,
    ) -> Tuple[TwistCommand, bool]:
        """
        Computes the velocity command to follow the path.
        :param current_pose: Current estimated pose of the UGV.
        :param path: Active planned path.
        :param delta_time: Time step in seconds.
        :return: (TwistCommand(linear_v, angular_w), goal_reached_boolean)
        """
        pass
