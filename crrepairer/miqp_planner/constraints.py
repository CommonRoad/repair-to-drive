import logging
import numpy as np
from typing import Union

import commonroad.common.validity as val
from commonroad.scenario.obstacle import DynamicObstacle
from crrepairer.miqp_planner.configuration import PlanningConfigurationVehicle

from crrepairer.smt.monitor_wrapper import PropositionNode

from crmonitor.common.vehicle import Vehicle

LOGGER = logging.getLogger(__name__)


class TIConstraints:

    def __init__(self, vehicle_configuration: PlanningConfigurationVehicle):

        self.a_max = vehicle_configuration.a_max

        # longitudinal constraints
        self.v_long_min = vehicle_configuration.v_lon_min
        self.v_long_max = vehicle_configuration.v_lon_max
        self.a_long_min = vehicle_configuration.a_lon_min
        self.a_long_max = vehicle_configuration.a_lon_max
        self.j_long_min = vehicle_configuration.j_lon_min
        self.j_long_max = vehicle_configuration.j_lon_max
        self.j_long_dot_min = vehicle_configuration.j_long_dot_min
        self.j_long_dot_max = vehicle_configuration.j_long_dot_max

        # lateral constraints
        self.kappa_dot_min = vehicle_configuration.kappa_dot_min
        self.kappa_dot_max = vehicle_configuration.kappa_dot_max
        self.kappa_dot_dot_min = vehicle_configuration.kappa_dot_dot_min
        self.kappa_dot_dot_max = vehicle_configuration.kappa_dot_dot_max
        self.kappa_max = vehicle_configuration.kappa_max

        self.theta_rel_min = vehicle_configuration.theta_rel_min
        self.theta_rel_max = vehicle_configuration.theta_rel_max

        self.react_time = vehicle_configuration.react_time
        self.wheelbase = vehicle_configuration.wheelbase
        self.wb_ra = vehicle_configuration.wb_ra
        self.wb_fa = vehicle_configuration.wb_fa
        self.length = vehicle_configuration.length


class LonConstraints:
    def __init__(self, N: int):
        self._valid_velocity = None
        assert N > 0
        self._v_min = np.repeat(-np.inf, N)
        self._v_max = np.repeat(np.inf, N)
        self._a_min = np.repeat(-np.inf, N)
        self._a_max = np.repeat(np.inf, N)
        self._preceding_vehicle = None
        self.tc_time_step = None
        self.s_soft_min = np.repeat(-np.inf, N)
        self.s_soft_max = np.repeat(np.inf, N)
        self.s_hard_min = np.repeat(-np.inf, N)
        self.s_hard_max = np.repeat(np.inf, N)
        self._select_proposition = None

    @classmethod
    def construct_constraints(
        cls,
        s_soft_min: np.ndarray,
        s_soft_max: np.ndarray,
        s_hard_min: np.ndarray,
        s_hard_max: np.ndarray,
        v_max=None,
        v_min=None,
        a_min=None,
        a_max=None,
        prec_veh=None,
        select_proposition=PropositionNode,
        tc_time_step=None,
    ):
        # check if constraints are valid vectors
        assert (
            val.is_real_number_vector(s_soft_min)
            and val.is_real_number_vector(s_soft_max)
            and val.is_real_number_vector(s_hard_min)
            and val.is_real_number_vector(s_hard_max)
        )
        # check if all constraint has the same length
        assert (
            len(s_soft_min) == len(s_soft_max)
            and len(s_soft_max) == len(s_hard_min)
            and len(s_hard_min) == len(s_hard_max)
        )

        constraints = LonConstraints(len(s_hard_max))
        constraints.s_soft_min = s_soft_min
        constraints.s_soft_max = s_soft_max
        constraints.s_hard_min = s_hard_min
        constraints.s_hard_max = s_hard_max

        # TODO: fix velocity constraints
        # v_max, v_min: [As, Av, b]: As * s + Av * v <= b
        no_arg = False
        if v_max is None or v_min is None:
            v_max = np.repeat(np.inf, len(s_soft_min))
            v_min = np.repeat(-np.inf, len(s_soft_min))
            no_arg = True
        assert len(v_max) == len(v_min) and len(v_max) == len(s_soft_min)

        if a_max is None or a_min is None:
            a_max = np.repeat(np.inf, len(s_soft_min))
            a_min = np.repeat(-np.inf, len(s_soft_min))
        assert len(a_max) == len(a_min) and len(a_max) == len(s_soft_min)

        constraints.v_max = v_max
        constraints.v_min = v_min
        constraints.a_min = a_min
        constraints.a_max = a_max
        constraints._valid_velocity = not no_arg
        constraints.preceding_vehicle = prec_veh
        constraints.tc_time_step = tc_time_step
        constraints.select_proposition = select_proposition
        return constraints

    @property
    def preceding_vehicle(self) -> Union[DynamicObstacle, Vehicle, None]:
        return self._preceding_vehicle

    @preceding_vehicle.setter
    def preceding_vehicle(self, pre_vehicle: Union[DynamicObstacle, Vehicle]):
        self._preceding_vehicle = pre_vehicle

    @property
    def valid_velocity(self) -> bool:
        return self._valid_velocity

    @property
    def select_proposition(self) -> PropositionNode:
        return self._select_proposition

    @select_proposition.setter
    def select_proposition(self, select_proposition: PropositionNode):
        self._select_proposition = select_proposition

    @property
    def v_max(self) -> np.ndarray:
        return self._v_max

    @v_max.setter
    def v_max(self, v_max: float):
        assert val.is_valid_velocity(v_max)
        self._v_max = v_max

    @property
    def v_min(self) -> np.ndarray:
        return self._v_min

    @v_min.setter
    def v_min(self, v_min: float):
        assert val.is_valid_velocity(v_min)
        self._v_min = v_min

    @property
    def a_min(self) -> float:
        return self._a_min

    @a_min.setter
    def a_min(self, a_min: float):
        assert val.is_valid_acceleration(a_min)
        self._a_min = a_min

    @property
    def a_max(self) -> float:
        return self._a_max

    @a_max.setter
    def a_max(self, a_max: float):
        assert val.is_valid_acceleration(a_max)
        self._a_max = a_max

    @property
    def s_soft_min(self) -> np.ndarray:
        return self._s_soft_min

    @s_soft_min.setter
    def s_soft_min(self, s_soft_min):
        assert val.is_real_number_vector(s_soft_min)
        self._s_soft_min = s_soft_min

    @property
    def s_soft_max(self) -> np.ndarray:
        return self._s_soft_max

    @s_soft_max.setter
    def s_soft_max(self, s_soft_max):
        assert val.is_real_number_vector(s_soft_max)
        self._s_soft_max = s_soft_max

    @property
    def s_hard_min(self) -> np.ndarray:
        return self._s_hard_min

    @s_hard_min.setter
    def s_hard_min(self, s_hard_min):
        assert val.is_real_number_vector(s_hard_min)
        self._s_hard_min = s_hard_min

    @property
    def s_hard_max(self) -> np.ndarray:
        return self._s_hard_max

    @s_hard_max.setter
    def s_hard_max(self, s_hard_max):
        assert val.is_real_number_vector(s_hard_max)
        self._s_hard_max = s_hard_max

    @property
    def N(self):
        return len(self.s_soft_min)

    @N.setter
    def N(self, N):
        pass


class LatConstraints:
    def __init__(self, N: int):
        assert N > 0
        self.d_soft_min = np.repeat(-np.inf, N)
        self.d_soft_max = np.repeat(np.inf, N)
        self.d_hard_min = np.repeat(-np.inf, N)
        self.d_hard_max = np.repeat(np.inf, N)

    @classmethod
    def construct_constraints(
        cls,
        d_soft_min: np.ndarray,
        d_soft_max: np.ndarray,
        d_hard_min: np.ndarray,
        d_hard_max: np.ndarray,
    ):
        # check if all constraint has the same length
        assert (
            len(d_soft_min) == len(d_soft_max)
            and len(d_soft_max) == len(d_hard_min)
            and len(d_hard_min) == len(d_hard_max)
        )

        lat_constraints = LatConstraints(len(d_soft_max))
        lat_constraints.d_soft_min = d_soft_min
        lat_constraints.d_soft_max = d_soft_max
        lat_constraints.d_hard_min = d_hard_min
        lat_constraints.d_hard_max = d_hard_max

        return lat_constraints

    @property
    def select_proposition(self) -> PropositionNode:
        return self._select_proposition

    @select_proposition.setter
    def select_proposition(self, select_proposition: PropositionNode):
        self._select_proposition = select_proposition

    @property
    def d_soft_min(self) -> list:
        return self._d_soft_min

    @d_soft_min.setter
    def d_soft_min(self, d_soft_min):
        assert (val.is_real_number_vector(d) for d in d_soft_min)
        self._d_soft_min = d_soft_min

    @property
    def d_soft_max(self) -> list:
        return self._d_soft_max

    @d_soft_max.setter
    def d_soft_max(self, d_soft_max):
        assert (val.is_real_number_vector(d) for d in d_soft_max)
        self._d_soft_max = d_soft_max

    @property
    def d_hard_min(self) -> list:
        return self._d_hard_min

    @d_hard_min.setter
    def d_hard_min(self, d_hard_min):
        assert (val.is_real_number_vector(d) for d in d_hard_min)
        self._d_hard_min = d_hard_min

    @property
    def d_hard_max(self):
        return self._d_hard_max

    @d_hard_max.setter
    def d_hard_max(self, d_hard_max):
        assert (val.is_real_number_vector(d) for d in d_hard_max)
        self._d_hard_max = d_hard_max

    @property
    def N(self):
        return len(self.d_hard_min)

    @N.setter
    def N(self, N):
        pass


class TVConstraints:
    """
    Time variant constraints
    """

    @classmethod
    def create_from_params(cls, horizon: float, N: int, dT: float):
        assert N == int(round(horizon / dT))
        lon, lat = cls.set_default_constraints(N)
        return cls(lon, lat)

    @staticmethod
    def set_default_constraints(N):
        lon = LonConstraints(N)
        lat = LatConstraints(N)

        return lon, lat

    def __init__(self, c_long: LonConstraints, c_lat: LatConstraints):
        self._N = c_long.N
        assert c_long.N == c_lat.N
        self._lon = c_long
        self._lat = c_lat

    @property
    def lon(self) -> LonConstraints:
        return self._lon

    @lon.setter
    def lon(self, c: LonConstraints):
        assert isinstance(c, LonConstraints) and c.N == self.N
        self._lon = c

    @property
    def lat(self) -> LatConstraints:
        return self._lat

    @lat.setter
    def lat(self, c: LatConstraints):
        assert isinstance(c, LatConstraints) and c.N == self.N
        self._lat = c

    @property
    def N(self) -> int:
        return self._N

    @N.setter
    def N(self, N: int):
        raise Exception("You are not allowed to change the horizon of the constraints!")
