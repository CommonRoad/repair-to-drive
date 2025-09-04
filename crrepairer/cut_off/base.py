import math
from abc import ABC, abstractmethod
from typing import List, Union
import matplotlib.pyplot as plt

# CommonRoad STL monitor
from crmonitor.common.world import World

# CommonRoad Toolbox
from commonroad.scenario.obstacle import DynamicObstacle, Shape
from commonroad.scenario.state import PMState, KSState, CustomState
import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
    create_collision_object,
)
import commonroad_dc.boundary.boundary as boundary
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_dc.collision.visualization.drawing import draw_collision_rectobb

from crrepairer.cut_off.utils import transfer_state_list_to_prediction


class CutOffBase(ABC):
    """
    Abstract base class for calculating cut-off states
    """

    def __init__(self, ego_vehicle: DynamicObstacle, world: World):
        self._world = world
        self.scenario = self._world.scenario
        self._ego_vehicle = ego_vehicle
        self._N = ego_vehicle.prediction.final_time_step
        self._dT = world.dt
        self._visualize = False
        if self.scenario.obstacle_by_id(self._ego_vehicle.obstacle_id) is not None:
            self.scenario.remove_obstacle(self._ego_vehicle)
        (
            road_boundary_obstacle,
            road_boundary_sg_rectangles,
        ) = boundary.create_road_boundary_obstacle(self.scenario)
        self.scenario.add_objects(road_boundary_obstacle)
        self._collision_checker = create_collision_checker(self.scenario)
        self.scenario.remove_obstacle(road_boundary_obstacle)
        if self._visualize:
            # visualize scenario and collision objects
            self.rnd = MPRenderer(figsize=(25, 10))
            self.scenario.lanelet_network.draw(self.rnd)
        # create the shape of the ego vehicle
        self._shape = self._ego_vehicle.obstacle_shape

    @property
    def world(self) -> World:
        return self._world

    @property
    def ego_vehicle(self) -> DynamicObstacle:
        return self._ego_vehicle

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def dT(self) -> float:
        return self._dT

    @dT.setter
    def dT(self, dT: float):
        raise Exception("You are not allowed to change the time step!")

    @property
    def N(self) -> int:
        return self._N

    @N.setter
    def N(self, N: int):
        raise Exception("You are not allowed to change the number of time steps!")

    @abstractmethod
    def generate(self, *args, **kwargs):
        """
        generates the cut off state: time-to-react or time-to-compliance
        """
        pass

    def _calc_ttc(self, state_list: List[Union[CustomState, PMState, KSState]]):
        """
        Detects the collision time given the trajectory of ego_vehicle using a for loop over
        the state list.
        """
        for i in range(len(state_list)):
            # ith time step
            pos1 = state_list[i].position[0]
            pos2 = state_list[i].position[1]
            theta = state_list[i].orientation
            # i: time_start_idx
            ego = pycrcc.TimeVariantCollisionObject(i)
            ego.append_obstacle(
                pycrcc.RectOBB(
                    0.5 * self._shape.length, 0.5 * self._shape.width, theta, pos1, pos2
                )
            )
            if self._collision_checker.collide(ego):
                if self._visualize:
                    rnd = MPRenderer()
                    ego_obb = pycrcc.RectOBB(
                        0.5 * self._shape.length,
                        0.5 * self._shape.width,
                        theta,
                        pos1,
                        pos2,
                    )
                    draw_collision_rectobb(ego_obb, rnd)
                    rnd.draw_params.time_begin = i
                    self.scenario.draw(rnd)
                    rnd.render()
                    plt.show()
                return i * self.dT
        return math.inf

    def _detect_collision(
        self, state_list: List[Union[CustomState, PMState, KSState]]
    ) -> bool:
        """
        return whether the state list of the ego vehicle is collision-free
        """
        # create a TrajectoryPrediction object consisting of the trajectory and the shape of the ego vehicle
        traj_pred = transfer_state_list_to_prediction(state_list, self._shape, self.dT)
        # create a collision object using the trajectory prediction of the ego vehicle
        co = create_collision_object(traj_pred)
        return self._collision_checker.collide(co)
