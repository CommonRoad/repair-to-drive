import warnings
from typing import Union, List, Any, Tuple
from collections import defaultdict
import math
import functools
from abc import ABC
import enum
import os
import copy
import numpy as np

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.state import CustomState, PMState, KSState

from commonroad_crime.utility.simulation import SimulationLong, SimulationLat, Maneuver
from commonroad_crime.data_structure.configuration import CriMeConfiguration
from commonroad_crime.utility.general import check_elements_state_list

from crrepairer.cut_off.base import CutOffBase
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.cut_off.utils import update_ego_vehicle, visualize_state_list, int_round


class TCSearchMode(str, enum.Enum):
    LINEAR = "linear search"
    BINARY = "binary search"


class TC(CutOffBase, ABC):
    """
    Time-To-Compliance.
    """

    def __init__(self, ego_vehicle: DynamicObstacle, rule_monitor: STLRuleMonitor):
        # avoid changing of rule_monitor in TSolver
        rule_monitor_copy = copy.copy(rule_monitor)
        rule_monitor_copy._world = copy.deepcopy(rule_monitor.world)
        # Create a list of vehicles to remove
        vehicles_to_remove = [
            veh
            for veh in rule_monitor_copy.world.vehicles
            if veh.id not in [ego_vehicle.obstacle_id, rule_monitor_copy.other_id]
        ]

        # Iterate over the list and remove each vehicle
        for veh in vehicles_to_remove:
            rule_monitor_copy.world.remove_vehicle(veh)
        super().__init__(ego_vehicle, rule_monitor_copy.world)

        # set round tolerance for different time step size
        if 0.01 <= self.dT < 0.1:
            self.round_tolerance = 2
        else:
            self.round_tolerance = 1
        self.rule_monitor = rule_monitor_copy
        self._world_ego = self.world.vehicle_by_id(ego_vehicle.obstacle_id)
        self._tv_time_step = self.rule_monitor.tv_time_step
        self._rule_to_other_id = self.rule_monitor.rule_to_other_id
        self._visualize = False
        self._save_state_lists = False

        self._compliant_maneuver = None
        self._tc = -math.inf
        self._tc_dict = defaultdict(float)
        self._mid = None
        self._search_mode = TCSearchMode.BINARY
        self.state_list_set = []

        # todo fix in params in crime
        yaml_file = os.path.join(
            os.getcwd(),
            "../../commonroad-criticality-measures/config_files/"
            + str(self.scenario.scenario_id)
            + ".yaml",
        )
        if os.path.exists(yaml_file):
            config = CriMeConfiguration.load(
                yaml_file,
                str(self.scenario.scenario_id),
            )
        else:
            config = CriMeConfiguration()
        config.scenario = self.scenario
        config.vehicle.cartesian.j_x_min = -5
        config.vehicle.cartesian.j_x_max = 5
        config.vehicle.cartesian.j_y_min = -5
        config.vehicle.cartesian.j_y_max = 5

        config.time.steer_width = 2  # use the lane width mode
        config.vehicle.ego_id = rule_monitor.vehicle_id

        self.config = config

        # simulators
        self._sim_lon = SimulationLong(
            Maneuver.NONE, copy.deepcopy(self.ego_vehicle), self.config
        )
        self._sim_lat = SimulationLat(
            Maneuver.NONE, copy.deepcopy(self.ego_vehicle), self.config
        )

    @property
    def simulation_lateral(self) -> Union[SimulationLat]:
        return self._sim_lat

    @property
    def simulation_longitudinal(self) -> Union[SimulationLong]:
        return self._sim_lon

    @property
    def tv(self):
        return int_round(self._tv_time_step * self.dT, self.round_tolerance)

    @property
    def future_time(self):
        return int_round(
            self.rule_monitor.future_time_step * self.dT, self.round_tolerance
        )

    @property
    def future_time_step(self):
        return self.rule_monitor.future_time_step

    @property
    def tc(self):
        if self._tc == -math.inf:
            return self._tc
        return int_round(self._tc, self.round_tolerance)

    @property
    def tc_time_step(self) -> Union[int, float]:
        if self._tc == -math.inf:
            return self._tc
        return int(self._tc / self.dT)

    @property
    def tv_time_step(self) -> Union[int, float]:
        return self._tv_time_step

    @property
    def compliant_maneuver(self) -> Maneuver:
        return self._compliant_maneuver

    def calc_tv_updated(
        self,
        updated_states: List[Union[CustomState, PMState, KSState]],
        cut_off_time: int,
        bool_mode,
    ) -> Tuple[float, Any]:
        # detect violation time using STL monitor
        # self.rule_monitor.evaluate_initially()
        self.rule_monitor.world.time_step = 0
        update_ego_vehicle(
            self.world.road_network, self._world_ego, updated_states, 0, self.dT
        )
        # TODO: FIXME future operator need be evaluated from start
        # cut_off_time_test = min(cut_off_time, self.tv_time_step - self.future_time_step)
        # rule_rob, other_ids = self.rule_monitor.evaluate_consecutively(
        #     self.world, cut_off_time_test
        # )
        rule_rob, other_ids = self.rule_monitor.evaluate_consecutively(
            self.world, self.rule_monitor.start_time_step, bool_mode
        )
        # check whether the rule_rob are of equal length, if not, should be a violation
        if not all(len(arr) == len(rule_rob[0]) for arr in rule_rob):
            return -math.inf, None
        else:
            rule_rob = np.array(rule_rob)
        if np.any(rule_rob[:, 0] < 0):
            rule_idx = np.where(rule_rob[:, 0] < 0)[0][0]
            if other_ids[rule_idx][0] == ():
                return -math.inf, None
            return -math.inf, other_ids[rule_idx][0][0]
        tv_per_rule = np.argmax(rule_rob < 0, axis=-1)
        if np.all(
            tv_per_rule + self._world_ego.start_time == self._world_ego.start_time
        ):
            return math.inf, None  # no violation
        min_tv = np.min(tv_per_rule[tv_per_rule != 0])
        rule_idx = np.where(tv_per_rule == min_tv)[0][0]
        if rule_idx == self.rule_monitor.min_rule_idx:
            if other_ids[rule_idx][min_tv] == ():
                return min_tv * self.dT, self.ego_vehicle.obstacle_id
            return min_tv * self.dT, other_ids[rule_idx][min_tv][0]
        else:
            print("Violated rule changed.")

    def generate(self, cut_off_maneuvers: List[Maneuver]):
        """
        Computes the Time-to-Compliance (with traffic rules).
        :param cut_off_maneuvers: the given maneuvers of ego vehicle
        :return: TC, corresponding maneuver
        """
        if not cut_off_maneuvers:
            return -math.inf
        if self.tv == -math.inf:
            raise ValueError(
                "<TC>: the trajectory is not repairable since it already disobeys the rules"
            )
        elif self.tv == math.inf:
            self._tc = math.inf
        else:
            ttm = dict()
            for maneuver in cut_off_maneuvers:
                if maneuver not in self._tc_dict:
                    if self._search_mode == TCSearchMode.BINARY:
                        ttm[maneuver] = self.search_ttm_binary(maneuver)
                    else:
                        ttm[maneuver] = self.search_ttm_linear(maneuver)
                    self._tc_dict[maneuver] = ttm[maneuver]
                else:
                    ttm[maneuver] = self._tc_dict[maneuver]

            self._tc = max(ttm.values())
            self._compliant_maneuver = max(ttm, key=ttm.get)
        return self._tc

    @functools.lru_cache(128)
    def search_ttm_binary(self, maneuver: Maneuver):
        ttm = -math.inf
        if self._tc_dict and max(self._tc_dict.values()) not in (math.inf, -math.inf):
            low = int(
                int_round(max(self._tc_dict.values()) / self.dT, self.round_tolerance)
            )
            if self.singleton_search(maneuver, low) == -math.inf:
                return -math.inf
            print(f"* \t<Tsolver>: {maneuver.value} start from the previous ttm {low}")
        else:
            low = self._world_ego.start_time
            print(f"* \t<Tsolver>: {maneuver.value} start from {low}")
        high = int(int_round(self.tv / self.dT, self.round_tolerance))
        while low < high:
            self._mid = int(int_round(low + high) / 2)
            tv = self.singleton_search(maneuver, self._mid)
            # if violation-free and collision-free
            if tv == math.inf:  # and not flag_collision:
                low = self._mid + 1
            else:
                high = self._mid

        if low != self._world_ego.start_time:
            ttm = (low - 1) * self.dT
        return ttm

    @functools.lru_cache(128)
    def search_ttm_linear(self, maneuver: Maneuver):
        ts = int(int_round(self.tv / self.dT, self.round_tolerance))
        while ts > 0:
            tv = self.singleton_search(maneuver, ts)
            if tv == math.inf:
                break
            else:
                ts -= 1
        if ts == 0:
            ttm = -math.inf
        else:
            ttm = ts * self.dT
        return ttm

    @functools.lru_cache(128)
    def singleton_search(self, maneuver: Maneuver, start_time: int):
        if maneuver in [Maneuver.BRAKE, Maneuver.KICKDOWN, Maneuver.CONSTANT]:
            self._sim_lon.update_maneuver(maneuver)
            state_list = self._sim_lon.simulate_state_list(start_time)
        elif maneuver in [Maneuver.STEERLEFT, Maneuver.STEERRIGHT]:
            self._sim_lat.update_maneuver(maneuver)
            state_list = self._sim_lat.simulate_state_list(start_time)
        else:
            raise ValueError(
                ": given compliant maneuver {} is not supported".format(maneuver)
            )
        if state_list is None:
            tv = -math.inf
        else:
            if self._visualize:
                visualize_state_list(
                    self._collision_checker,
                    state_list,
                    self.scenario,
                    self._sim_lat.vehicle_dynamics.shape,
                )
            if self._save_state_lists:
                self.state_list_set.append(state_list[start_time:])
            check_elements_state_list(state_list, self.dT)
            try:
                tv, _ = self.calc_tv_updated(
                    state_list, self._mid, True
                )  # which should be tv instead of ttm
            except AttributeError as e:
                # Warn the user about the attribute error
                warnings.warn(f"* \t<Tsolver>: AttributeError encountered: {e}")
                tv = -math.inf
        return tv
