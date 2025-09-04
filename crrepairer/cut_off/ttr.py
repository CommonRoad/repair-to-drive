from abc import ABC
import math
import functools
import os

from crrepairer.cut_off.base import CutOffBase
from crrepairer.cut_off.utils import visualize_state_list, int_round

from commonroad_crime.utility.simulation import SimulationLong, SimulationLat, Maneuver
from commonroad_crime.utility.general import check_elements_state_list
from commonroad_crime.data_structure.configuration import CriMeConfiguration

from crmonitor.common.world import World


class TTR(CutOffBase, ABC):
    """
    Time-To-React.
    """

    def __init__(self, ego_obstacle, world: World):
        super().__init__(ego_obstacle, world)
        ego_vehicle = world.vehicle_by_id(ego_obstacle.obstacle_id)
        # calculate the time-to-collision as default value
        self._ttc = self._calc_ttc(ego_vehicle.states_cr)
        self._visualize = False

        self.config = ConfigurationBuilder.build_configuration(
            str(self.scenario.scenario_id)
        )
        self.config.scenario = self.scenario

    @property
    def ttc(self):
        return self._ttc

    def generate(self, emergency_maneuvers):
        """
        Computes the time-to-react(TTR).
        :param emergency_maneuvers: the given set of emergency maneuvers
        :return: TTR, corresponding maneuver
        """
        # time to execute certain evasive maneuver
        ttm = dict()
        if self._ttc == 0:
            return -math.inf
        elif self._ttc == math.inf:
            return math.inf
        else:
            for maneuver in emergency_maneuvers:
                ttm[maneuver] = self.search_ttm(maneuver)
            if max(ttm.values()) in [math.inf, -math.inf]:
                return max(ttm.values())
            return int_round(max(ttm.values()), 1)  # , max(ttm, key=ttm.get)

    @functools.lru_cache(128)
    def search_ttm(self, maneuver):
        """
        Finds the TTM.
        """
        ttm = -math.inf
        low = 0
        high = int(self._ttc / self.dT)
        while low < high:
            mid = int((low + high) / 2)
            if maneuver in [Maneuver.BRAKE, Maneuver.KICKDOWN, Maneuver.CONSTANT]:
                SL = SimulationLong(maneuver, self.ego_vehicle, self.config)
            elif maneuver in [Maneuver.STEERLEFT, Maneuver.STEERRIGHT]:
                SL = SimulationLat(maneuver, self.ego_vehicle, self.config)
            else:
                raise ValueError(
                    "<TTR>: given compliant maneuver {} is not supported".format(
                        maneuver
                    )
                )
            state_list = SL.simulate_state_list(mid)
            check_elements_state_list(state_list, self.dT)
            if state_list is None:
                return -math.inf
            if self._visualize:
                visualize_state_list(
                    self._collision_checker,
                    state_list,
                    self.scenario,
                    SL.vehicle_dynamics.shape,
                )
            flag_collision = self._detect_collision(state_list)  # bool value
            # if collision-free
            if not flag_collision:
                low = mid + 1
            else:
                high = mid
        if low != 0:
            ttm = (low - 1) * self.dT
        return ttm
