import enum
import dataclasses
import inspect
import os.path
from dataclasses import dataclass, field
from typing import Union, Any, Optional, Dict, List
import pathlib
from omegaconf import OmegaConf

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet

from crrepairer.utils.general import load_scenario_and_planning_problem

from crrepairer.miqp_planner.configuration import ConfigurationBuilder


class ReferencePoint(enum.Enum):
    """
    The reference point for the ego vehicle to describe the kinematics of the ego vehicle.
    If we choose the rear axis, the slip angle can be assumed to be zero at low speeds and generally
    neglected at high speeds.
    """

    CENTER = "center"
    REAR = "rear"


class ScenarioType(str, enum.Enum):
    """
    Type of scenario used for repairing.
    """

    INTERSTATE = "interstate"
    INTERSECTION = "intersection"


class MonitorType(enum.Enum):
    """
    Type of temporal logic used in the traffic rule monitor
    """

    MTL = "metric temporal logic"
    STL = "signal temporal logic"


class IntersectionType(str, enum.Enum):
    HAND_DRAFT = "hand_draft"
    DATASET = "dataset"


def _dict_to_params(dict_params: Dict[str, Any], cls: Any) -> Any:
    """
    Converts dictionary to parameter class.

    :param dict_params: Dictionary containing parameters.
    :param cls: Parameter dataclass to which dictionary should be converted to.
    :return: Parameter class.
    """
    fields = dataclasses.fields(cls)
    cls_map = {f.name: f.type for f in fields}
    kwargs = {}
    for k, v in cls_map.items():
        if k not in dict_params:
            continue
        if inspect.isclass(v) and issubclass(v, BaseConfiguration):
            kwargs[k] = _dict_to_params(dict_params[k], cls_map[k])
        else:
            kwargs[k] = dict_params[k]
    return cls(**kwargs)


@dataclass
class BaseConfiguration:
    """Reactive planner base parameters."""

    __initialized: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        """Post initialization of base parameter class."""
        # pylint: disable=unused-private-member
        self.__initialized = True
        # Make sure that the base parameters are propagated to all sub-parameters
        # This cannot be done in the init method, because the sub-parameters are not yet initialized.
        # This is not a noop, as it calls the __setattr__ method.
        # Do not remove!
        # See commonroad-io how to set the base parameters

    def __getitem__(self, item: str) -> Any:
        """
        Getter for base parameter value.

        :param: Item for which content should be returned.
        :return: Item value.
        """
        try:
            value = self.__getattribute__(item)
        except AttributeError as e:
            raise KeyError(
                f"{item} is not a parameter of {self.__class__.__name__}"
            ) from e
        return value

    def __setitem__(self, key: str, value: Any):
        """
        Setter for item.

        :param key: Name of item.
        :param value: Value of item.
        """
        try:
            self.__setattr__(key, value)
        except AttributeError as e:
            raise KeyError(
                f"{key} is not a parameter of {self.__class__.__name__}"
            ) from e

    @classmethod
    def load(
        cls,
        file_path: Union[pathlib.Path, str],
        scenario_name: str,
        validate_types: bool = True,
    ) -> "RepairerConfiguration":
        """
        Loads config file and creates parameter class.

        :param file_path: Path to yaml file containing config parameters.
        :param scenario_name: Name of scenario which should be used.
        :param validate_types:  Boolean indicating whether loaded config should be validated against CARLA parameters.
        :return: Base parameter class.
        """
        file_path = pathlib.Path(file_path)
        assert (
            file_path.suffix == ".yaml"
        ), f"File type {file_path.suffix} is unsupported! Please use .yaml!"
        loaded_yaml = OmegaConf.load(file_path)
        if validate_types:
            OmegaConf.merge(OmegaConf.structured(RepairerConfiguration), loaded_yaml)
        params = _dict_to_params(OmegaConf.to_object(loaded_yaml), cls)
        params.general.set_path_scenario(scenario_name + ".xml")
        return params


@dataclass
class RepairConfiguration(BaseConfiguration):
    """Detailed parameters for repairer."""

    # constraint mode
    constraint_mode: int = 1  # 1: Manual, 2: Reach

    # the id of the vehicle, whose trajectory needs to be repaired
    ego_id: int = 201

    rules: List[str] = field(default_factory=lambda: ["R_G1"])

    # initial time step
    t_0: int = 0
    # number of time steps
    N_r: int = 21

    # type of scenario, affecting the map and rule monitoring
    scenario_type: str = ScenarioType.INTERSTATE
    # type of intersection: hand-crafted or from dataset
    intersection_type: str = IntersectionType.HAND_DRAFT

    multiproc: bool = True
    use_mpr: bool = False

    use_mpr_derivative: bool = False

    def __post_init__(self):
        pass

    @property
    def t_f(self) -> int:
        return self.t_0 + self.N_r


@dataclass
class MIQPPlannerConfiguration(BaseConfiguration):
    """Parameters for MIQP planner."""

    # time horizon for the MIQP planning
    horizon: Optional[float] = None
    # nr of time steps
    N_p: Optional[int] = None

    slack_long: bool = True
    slack_lat: bool = True
    # s, v, a, j, u, slack
    weight_long: List[float] = field(
        default_factory=lambda: [0.1, 0.1, 0.5, 1, 0.1, 1000000]
    )
    # d, theta, kappa, kappa_dot, u, robust, slack
    weight_lat: List[float] = field(
        default_factory=lambda: [0.1, 0.1, 0.5, 1.0, 0.1, 0.1, 1000000]
    )

    def __post_init__(self):
        pass


@dataclass
class DebugConfiguration(BaseConfiguration):
    """Parameters specifying debug-related information."""

    # save plots
    save_plots: bool = False
    # save config/logs
    save_config: bool = False
    # show plots
    show_plots: bool = False
    # draw the reference path
    draw_ref_path: bool = True
    # draw the planning problem
    draw_planning_problem: bool = True
    # draw obstacles with vehicle icons
    draw_icons: bool = False
    # logging settings - Options: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging_level: str = "INFO"
    # use multiprocessing True/False
    multiproc: bool = True
    # number of workers for multiprocessing
    num_workers: int = 6

    # plotting limits
    plot_limits: Optional[List] = None


@dataclass
class VehicleConfiguration(BaseConfiguration):
    """Class to store vehicle configurations"""

    config_builder = ConfigurationBuilder()
    config_builder.set_root_path(
        root=os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../miqp_planner/")
        ),
        path_to_config="configurations",
    )
    qp_veh_config = config_builder.build_configuration()

    kappa_dot_dot_min: float = -100
    kappa_dot_dot_max: float = 100
    kappa_dot_min: float = -0.4
    kappa_dot_max: float = 0.4
    kappa_min: float = -0.5
    kappa_max: float = 0.5


@dataclass
class GeneralConfiguration(BaseConfiguration):
    """General parameters for evaluations."""

    # paths are relative to the root directory
    path_root_abs = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
    path_scenarios: str = path_root_abs + "/scenarios/"
    path_output: str = path_root_abs + "/output/"
    path_logs: str = path_root_abs + "/output/logs/"
    path_figures: str = path_root_abs + "/output/figures/"

    path_scenario: Optional[str] = None
    name_scenario: Optional[str] = None

    def __post_init__(self):
        # it will not throw an error if the directory already exists
        os.makedirs(self.path_output, exist_ok=True)
        os.makedirs(self.path_logs, exist_ok=True)
        os.makedirs(self.path_figures, exist_ok=True)

    def set_path_scenario(self, scenario_name: str):
        """
        Setter for scenario path.

        :param scenario_name: Name of CommonRoad scenario.
        """
        if not scenario_name.endswith(".xml"):
            self.path_figures = os.path.join(self.path_figures, scenario_name + ".xml")
            self.path_scenario = os.path.join(
                self.path_scenarios, scenario_name + ".xml"
            )
        else:
            self.path_figures = os.path.join(self.path_figures, scenario_name)
            self.path_scenario = os.path.join(self.path_scenarios, scenario_name)

        os.makedirs(self.path_figures, exist_ok=True)


@dataclass
class RepairerConfiguration(BaseConfiguration):
    """Configuration parameters for trajectory repairer."""

    vehicle: VehicleConfiguration = field(default_factory=VehicleConfiguration)
    debug: DebugConfiguration = field(default_factory=DebugConfiguration)
    general: GeneralConfiguration = field(default_factory=GeneralConfiguration)
    miqp_planner: MIQPPlannerConfiguration = field(
        default_factory=MIQPPlannerConfiguration
    )
    repair: RepairConfiguration = field(default_factory=RepairConfiguration)

    def __post_init__(self):
        self.scenario: Optional[Scenario] = None
        self.planning_problem: Optional[PlanningProblem] = None
        self.planning_problem_set: Optional[PlanningProblemSet] = None

    @property
    def name_scenario(self) -> str:
        return self.general.name_scenario

    def update(
        self, scenario: Scenario = None, planning_problem: PlanningProblem = None
    ):
        """
        Updates configuration based on the given attributes.
        Function used to construct initial configuration before planner initialization and update configuration during
        re-planning.

        :param scenario: (initial or updated) Scenario object
        :param planning_problem: (initial or updated) planning problem
        """
        # update scenario and planning problem with explicitly given ones
        if scenario:
            self.scenario = scenario
        if planning_problem:
            self.planning_problem = planning_problem

        # if scenario and planning problem not explicitly given
        if scenario is None and planning_problem is None:
            if self.scenario is None or self.planning_problem is None:
                # read original scenario and pp from scenario file
                self.scenario, self.planning_problem, self.planning_problem_set = (
                    load_scenario_and_planning_problem(self.general.path_scenario)
                )
            else:
                # keep previously stored scenario and planning problem
                pass
        else:
            pass

        # Check that scenario and planning problem are set
        assert (
            self.scenario is not None
        ), "<Configuration.update()>: no scenario has been specified"
        assert (
            self.planning_problem is not None
        ), "<Configuration.update()>: no planning problem has been specified"

        # Complete the scenario name
        # self.general.name_scenario = str(self.scenario.scenario_id)
