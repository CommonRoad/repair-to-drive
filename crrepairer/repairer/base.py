from abc import ABC, abstractmethod
from commonroad.scenario.trajectory import Trajectory

"""
Base c lass for trajectory repairer.
"""


class TrajectoryRepair(ABC):
    """
    Abstract base class for a trajectory repairer.
    Contains basic methods and properties every repairer has to offer.
    """

    def __init__(
        self,
        initial_trajectory: Trajectory,
    ):
        self._initial_trajectory = initial_trajectory

    @property
    def initial_trajectory(self) -> Trajectory:
        return self._initial_trajectory

    @initial_trajectory.setter
    def initial_trajectory(self, initial_trajectory: Trajectory):
        raise Exception(
            "You are not allowed to change the initial trajectory of the repairer!"
        )

    @abstractmethod
    def repair(self, *args, **kwargs):
        """
        Repair the trajectory
        :return: The repaired trajectory based on the needs
        """
        pass
