import os
import enum
import glob
import logging
import numpy as np
from typing import Union
from omegaconf import ListConfig, DictConfig, OmegaConf

# commonroad-io
from vehiclemodels.vehicle_parameters import VehicleParameters

from commonroad.common.validity import is_real_number
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.common.solution import VehicleType
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping

# commonroad-curvilinear-coordinate-system
import commonroad_clcs.pycrccosy as pycrccosy
from commonroad_clcs.util import (
    compute_orientation_from_polyline,
    compute_curvature_from_polyline,
    compute_pathlength_from_polyline,
)

LOGGER = logging.getLogger(__name__)

VEHICLE_ID = int
TIME_IDX = int


class ReferencePoint(enum.Enum):
    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member

    """
    The reference point for the ego vehicle to describe the kinematics of the ego vehicle.
    If we choose the rear axis, the slip angle can be assumed to be zero at low speeds and generally
    neglected at high speeds.
    """
    CENTER = "center"
    REAR = "rear"


class LinearVehicleModel(enum.Enum):
    """
    The QP planner is capable of planning for linear vehicle models. This enum class provide the way to
    change the state transition matrices in the longitudinal and/or lateral planner.
    """

    QP_VEHICLE_MODEL = "QP_VEHICLE_MODEL"
    LINEARIZED_KS_VEHICLE_MODEL = "LINEARIZED_KS_VEHICLE_MODEL"


class PlanningConfigurationVehicle:
    """Class which holds all necessary vehicle-specific parameters for the
    trajectory planning."""

    def __init__(
        self, config: Union[ListConfig, DictConfig], logger_level=logging.INFO
    ):
        """
        Default settings.
        """
        config_relevant = config.vehicle.ego

        LOGGER.setLevel(logger_level)
        # 1 = Ford Escord, 2 = BMW 320i, 3 = VW Vanagon, 4 = Semi-trailer truck
        self.id_type_vehicle = config_relevant.id_type_vehicle
        self.vehicle_id = -1

        p = VehicleParameterMapping.from_vehicle_type(VehicleType(self.id_type_vehicle))

        self.v_lon_min = p.longitudinal.v_min
        self.v_lon_max = p.longitudinal.v_max

        self.a_lon_min = -p.longitudinal.a_max
        self.a_lon_max = p.longitudinal.a_max

        self.a_max = p.longitudinal.a_max

        # TODO: extend from current VehicleParameters
        # j = (a_max - a_min) / dt = 23 / 0.04 = 575
        self.j_lon_min = -1000.0  # 1000.0
        self.j_lon_max = 1000.0  # 1000.0
        self.j_long_dot_min = -1000
        self.j_long_dot_max = 1000

        # lateral constraints
        self.kappa_dot_min = -0.4  # minimum steering rate
        self.kappa_dot_max = 0.4  # maximum steering rate
        self.kappa_dot_dot_min = -20.0  # minimum steering rate rate
        self.kappa_dot_dot_max = 20.0  # maximum steering rate rate
        self.kappa_max = 0.20  # maximum curvature

        self.theta_rel_min = (
            -0.1408
        )  # linearization of vehicle model relys on small angle approximation
        self.theta_rel_max = 0.1408
        self.react_time = 0.4

        # self._desired_speed = 0.0  # not used

        self.wb_fa = p.a
        self.wb_ra = p.b
        self.wheelbase = self.wb_fa + self.wb_ra
        self.length = p.l
        self.width = p.w

        # overwrite with parameters given by vehicle ID if they are explicitly provided in the *.yaml file
        for key, value in config_relevant.items():
            if value is not None:
                setattr(self, key, value)

        self._curvilinear_coordinate_system = None
        self._reference_path = None
        self._reference_point = ReferencePoint(config.planning.reference_point)
        self._lanelet_network = None

        self.ref_pos = None
        self.ref_theta = None
        self.ref_curv = None

    @property
    def CLCS(self) -> pycrccosy.CurvilinearCoordinateSystem:
        """Curvilinear coordinate system for the reachable set computation in curvilinear coordinates."""
        if (
            self._curvilinear_coordinate_system is None
            and self.reference_path is not None
        ):
            self._curvilinear_coordinate_system = pycrccosy.CurvilinearCoordinateSystem(
                self.reference_path
            )
        return self._curvilinear_coordinate_system

    @CLCS.setter
    def CLCS(
        self, curvilinear_coordinate_system: pycrccosy.CurvilinearCoordinateSystem
    ):
        # assert (isinstance(curvilinear_coordinate_system, pycrccosy.CurvilinearCoordinateSystem)), \
        #     '<PlanConfiguration/curvilinear_coordinate_system> Expected type ' \
        #     'pycrccosy.PolylineCoordinateSystem; Got type %s instead.' % (type(curvilinear_coordinate_system))
        self._curvilinear_coordinate_system = curvilinear_coordinate_system

    @property
    def reference_path(self):
        """Reference path of the vehicle for the generation of a curvilinear coordinate system or trajectory
        planning. The reference path must be given as polyline."""
        return self._reference_path

    @reference_path.setter
    def reference_path(self, reference_path: np.ndarray):
        if reference_path is not None:
            assert (
                isinstance(reference_path, np.ndarray)
                and reference_path.ndim == 2
                and len(reference_path) > 1
                and len(reference_path[0, :]) == 2
            ), "<PlanConfiguration/reference_path>: Provided reference is not valid. reference = {}".format(
                reference_path
            )
            self._reference_path = reference_path
            if not hasattr(self.CLCS, "ref_pos"):
                self.ref_pos = compute_pathlength_from_polyline(reference_path)
            else:
                self.ref_pos = self.CLCS.ref_pos

            if not hasattr(self.CLCS, "ref_theta"):
                self.ref_theta = compute_orientation_from_polyline(reference_path)
            else:
                self.ref_theta = self.CLCS.ref_theta

            if not hasattr(self.CLCS, "ref_pos"):
                self.ref_curv = compute_curvature_from_polyline(reference_path)
            else:
                self.ref_curv = self.CLCS.ref_curv

    @property
    def reference_point(self) -> ReferencePoint:
        return self._reference_point

    @reference_point.setter
    def reference_point(self, reference_point: Union[str, ReferencePoint]):
        if isinstance(reference_point, str):
            reference_point = ReferencePoint(reference_point)
        assert isinstance(reference_point, ReferencePoint), (
            "<ReachSetConfiguration/reference_point>: argument reference_point of wrong type. Expected type: %s. "
            "Got type: %s" % (ReferencePoint, type(reference_point))
        )
        self._reference_point = reference_point

    @property
    def lanelet_network(self) -> Union[None, LaneletNetwork]:
        """The part of the lanelet network of the scenario, the vehicle is allowed or should drive on."""
        return self._lanelet_network

    @lanelet_network.setter
    def lanelet_network(self, lanelet_network: LaneletNetwork):
        assert isinstance(lanelet_network, LaneletNetwork), (
            "<PlanConfiguration/lanelet_network>: argument "
            "lanelet_network of wrong type. Expected type: %s. "
            "Got type: %s." % (LaneletNetwork, type(lanelet_network))
        )
        self._lanelet_network = lanelet_network


class ConfigurationBuilder:
    path_root: str = None
    path_config: str = None
    path_config_default: str = None
    dict_config_overridden: dict = None

    @classmethod
    def set_root_path(
        cls,
        root: str,
        path_to_config: str = "configurations",
        dir_configs_default: str = "defaults",
    ):
        """Sets the path to the root directory.

        Args:
            root (str): root directory
            path_to_config (str): relative path of configurations to root path
            dir_configs_default (str): directory under root folder containing
            default config files.
        """

        assert os.path.exists(root)

        cls.path_root = root
        cls.path_config = os.path.join(root, path_to_config)
        assert os.path.exists(cls.path_config)

        cls.path_config_default = os.path.join(cls.path_config, dir_configs_default)
        assert os.path.exists(cls.path_config_default)

    @classmethod
    def build_configuration(cls, name_scenario=None) -> PlanningConfigurationVehicle:
        """Builds configuration from default and scenario-specific config files.

        Steps:
            1. Load default config files
            2. Override if scenario-specific config file exists
            3. Build Configuration object
            4. Load scenario and planning problems
            5. Complete configuration with scenario and planning problem

        Args:
            scenario_loader (ScenarioLoader): scenario loader object
            scenario_id (str): considered scenario
            idx_planning_problem (int, optional): index of the planning problem.
            Defaults to 0.

        Returns:
            CommonRoadISSConfigurations: configuration containing all relevant information
        """
        config_default = cls.construct_default_configuration()
        config_cli = OmegaConf.from_cli()
        if name_scenario:
            config_scenario = cls.construct_scenario_configuration(name_scenario)

            # configurations coming after overrides the ones coming before
            config_combined = OmegaConf.merge(
                config_default, config_scenario, config_cli
            )
        else:
            config_combined = OmegaConf.merge(config_default, config_cli)

        # 3. Build Configuration object
        config = PlanningConfigurationVehicle(config_combined)

        # 4. Load scenario and planning problems

        # TODO remove this part
        # 5. Complete configuration with scenario and planning problem
        # config.complete_configuration(scenario, planning_problem)
        # config.planning_problem_idx = planning_problem_id

        return config

    @classmethod
    def construct_default_configuration(cls) -> Union[ListConfig, DictConfig]:
        """Constructs default configuration by accumulating yaml files.

        Collects all configuration files ending with '.yaml'.
        """
        config_default = OmegaConf.create()
        for path_file in glob.glob(cls.path_config_default + "/*.yaml"):
            with open(path_file, "r") as file_config:
                try:
                    config_partial = OmegaConf.load(file_config)
                    name_file = path_file.split("/")[-1].split(".")[0]

                except Exception as e:
                    print(e)

                else:
                    config_default[name_file] = config_partial

        config_default = cls.convert_to_absolute_paths(config_default)

        return config_default

    @classmethod
    def convert_to_absolute_paths(
        cls, config_default: Union[ListConfig, DictConfig]
    ) -> Union[ListConfig, DictConfig]:
        """Converts relative paths to absolute paths."""
        for key, path in config_default["general"].items():
            path_relative = os.path.join(cls.path_root, path)
            if os.path.exists(path_relative):
                config_default["general"][key] = path_relative

        config_default["general"]["path_root"] = cls.path_root

        return config_default

    @classmethod
    def construct_scenario_configuration(
        cls, name_scenario: str
    ) -> Union[DictConfig, ListConfig]:
        """
        Constructs scenario-specific configuration.

        """
        config_scenario = OmegaConf.create()

        path_config_scenario = cls.path_config + f"/{name_scenario}.yaml"
        if os.path.exists(path_config_scenario):
            with open(path_config_scenario, "r") as file_config:
                try:
                    config_scenario = OmegaConf.load(file_config)

                except Exception as e:
                    print(e)

        # add scenario name to the config file
        config_scenario["general"] = {"name_scenario": name_scenario}

        return config_scenario
