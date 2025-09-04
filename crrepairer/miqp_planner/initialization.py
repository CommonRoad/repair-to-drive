from typing import Dict, Tuple, List, Union, Any, Optional
import numpy as np
import logging
import os

from commonroad_route_planner.route_planner import RoutePlanner

# commonroad-io
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import State, InitialState, CustomState, KSState, PMState
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.common.util import make_valid_orientation

# commonroad-curvilinear-coordinatesystem
import commonroad_clcs.pycrccosy as pycrccosy
from commonroad_clcs.pycrccosy import CurvilinearCoordinateSystem
from commonroad_clcs.util import (
    chaikins_corner_cutting,
    resample_polyline,
    compute_pathlength_from_polyline,
    compute_orientation_from_polyline,
    compute_curvature_from_polyline,
)

from crrepairer.miqp_planner.trajectory import TrajPoint
from crrepairer.miqp_planner.configuration import (
    PlanningConfigurationVehicle,
    ReferencePoint,
    ConfigurationBuilder,
)
from crrepairer.miqp_planner.utils import validate_orientation
from crrepairer.smt.monitor_wrapper import ScenarioType
from crrepairer.utils.general import create_curvilinear_coordinate_system

from crmonitor.common.world import DynamicObstacleVehicle, Vehicle


LOGGER = logging.getLogger(__name__)


def set_up_miqp(
    settings: Dict,
    scenario: Scenario,
    planning_problem: PlanningProblem,
    vehicle: Optional[DynamicObstacleVehicle],
):
    """
    create vehicle configuration for the optimization problem
    """
    vehicle_configuration = create_optimization_configuration_vehicle(
        scenario,
        planning_problem,
        settings["vehicle_settings"],
        settings["scenario_type"],
        vehicle,
    )
    return vehicle_configuration


def create_optimization_configuration_vehicle(
    scenario: Scenario,
    planning_problem: PlanningProblem,
    settings: Dict,
    scenario_type: ScenarioType,
    vehicle: DynamicObstacleVehicle,
    path_root_configs: str = None,
    path_to_config: str = "configurations",
):
    vehicle_settings = settings[planning_problem.planning_problem_id]

    LOGGER.warning("vehicle parameters will be overwritten by vehicle model")
    config_builder = ConfigurationBuilder()
    if path_root_configs:
        config_builder.set_root_path(
            root=path_root_configs, path_to_config=path_to_config
        )
    else:
        config_builder.set_root_path(
            root=os.path.normpath(os.path.join(os.path.dirname(__file__))),
            path_to_config=path_to_config,
        )
    configuration = config_builder.build_configuration(
        name_scenario=str(scenario.scenario_id)
    )

    if scenario_type == "intersection":
        # use reference path and lanelets_dir from rule monitor
        reference_path = vehicle.ref_path_lane
        lanelets_leading_to_goal = vehicle.lanelets_dir
        configuration.reference_path = reference_path.smoothed_vertices
        configuration.curvilinear_coordinate_system = reference_path.clcs
    else:
        route_planner = RoutePlanner(scenario.lanelet_network, planning_problem)
        (
            reference_path,
            lanelets_leading_to_goal,
        ) = find_reference_path_and_lanelets_leading_to_goal(
            route_planner, planning_problem, settings
        )
        configuration.reference_path = np.array(reference_path)
        configuration.curvilinear_coordinate_system = (
            create_curvilinear_coordinate_system(configuration.reference_path)
        )
    configuration.lanelet_network = create_lanelet_network(
        scenario.lanelet_network, lanelets_leading_to_goal
    )

    if "reference_point" in vehicle_settings:
        configuration.reference_point = set_reference_point(
            vehicle_settings["reference_point"]
        )
    configuration.vehicle_id = planning_problem.planning_problem_id
    configuration.min_speed_x = vehicle_settings["min_speed_x"]
    configuration.max_speed_x = vehicle_settings["max_speed_x"]
    configuration.min_speed_y = vehicle_settings["min_speed_y"]
    configuration.max_speed_y = vehicle_settings["max_speed_y"]

    configuration.a_max_x = vehicle_settings["a_max_x"]
    configuration.a_min_x = vehicle_settings["a_min_x"]
    configuration.a_max_y = vehicle_settings["a_max_y"]
    configuration.a_min_y = vehicle_settings["a_min_y"]
    configuration.a_max = vehicle_settings["a_max"]

    configuration.j_min_x = vehicle_settings["j_min_x"]
    configuration.j_max_x = vehicle_settings["j_max_x"]
    configuration.j_min_y = vehicle_settings["j_min_y"]
    configuration.j_max_y = vehicle_settings["j_max_y"]

    configuration.length = vehicle_settings["length"]
    configuration.width = vehicle_settings["width"]
    configuration.wheelbase = vehicle_settings["wheelbase"]
    configuration.react_time = vehicle_settings["react_time"]
    configuration.radius, _ = compute_approximating_circle_radius(
        configuration.length, configuration.width
    )

    return configuration


def set_reference_point(reference_point: str) -> ReferencePoint:
    if reference_point == "rear":
        return ReferencePoint.REAR
    elif reference_point == "center":
        return ReferencePoint.CENTER
    else:
        raise ValueError(
            "<set_reference_point>: reference point of the ego vehicle is unknown: {}".format(
                reference_point
            )
        )


def find_reference_path_and_lanelets_leading_to_goal(
    route_planner: RoutePlanner, planning_problem: PlanningProblem, settings: Dict
):
    """
    Find a reference path (and the corresponding lanelets) to the given goal region. The obtained reference path will be
    resampled if needed.
    """

    def interval_extract(list_ids):
        list_ids = sorted(set(list_ids))
        range_start = previous_number = list_ids[0]
        merged_intervals = list()
        for number in list_ids[1:]:
            if number == previous_number + 1:
                previous_number = number
            else:
                merged_intervals.append([range_start, previous_number])
                range_start = previous_number = number
        merged_intervals.append([range_start, previous_number])
        return merged_intervals

    vehicle_settings = settings[planning_problem.planning_problem_id]

    routes = route_planner.plan_routes()
    lanelets_leading_to_goal = routes.retrieve_first_route().lanelet_ids
    reference_path = routes.retrieve_first_route().reference_path

    # visualize the route
    # visualize_route(route, draw_route_lanelets=True, draw_reference_path=True, size_x=6)

    # extend the reference path:
    first_lanelet = route_planner.lanelet_network.find_lanelet_by_id(
        lanelets_leading_to_goal[0]
    )
    while first_lanelet.predecessor:
        first_lanelet = route_planner.lanelet_network.find_lanelet_by_id(
            first_lanelet.predecessor[0]
        )
        reference_path = np.concatenate((first_lanelet.center_vertices, reference_path))
    last_lanelet = route_planner.lanelet_network.find_lanelet_by_id(
        lanelets_leading_to_goal[-1]
    )
    while last_lanelet.successor:
        last_lanelet = route_planner.lanelet_network.find_lanelet_by_id(
            last_lanelet.successor[0]
        )
        reference_path = np.concatenate((reference_path, last_lanelet.center_vertices))

    max_curvature = vehicle_settings["max_curvature_reference_path"] + 0.2
    # resampling the reference path
    if vehicle_settings["resampling_reference_path"]:
        while max_curvature > vehicle_settings["max_curvature_reference_path"]:
            reference_path = np.array(chaikins_corner_cutting(reference_path))
            reference_path = resample_polyline(
                reference_path, vehicle_settings["resampling_reference_path"]
            )
            abs_curvature = abs(compute_curvature_from_polyline(reference_path))
            max_curvature = max(abs_curvature)
        if (
            "resampling_reference_path_depending_on_curvature" in vehicle_settings
            and vehicle_settings["resampling_reference_path_depending_on_curvature"]
        ):
            # resample path with higher value where curvature is small
            resampled_path = list()
            intervals = list()
            abs_curvature[0:5] = 0.2
            merged_intervals_ids = interval_extract(
                [i for i, v in enumerate(abs_curvature) if v < 0.01]
            )
            for i in range(0, len(merged_intervals_ids) - 1):
                if i == 0 and merged_intervals_ids[i][0] != 0:
                    intervals.append([0, merged_intervals_ids[i][0]])
                if merged_intervals_ids[i][0] != merged_intervals_ids[i][1]:
                    intervals.append(merged_intervals_ids[i])
                intervals.append(
                    [merged_intervals_ids[i][1], merged_intervals_ids[i + 1][0]]
                )

            if len(merged_intervals_ids) == 1:
                if merged_intervals_ids[0][0] != 0:
                    intervals.append([0, merged_intervals_ids[0][0]])
                if merged_intervals_ids[0][0] != merged_intervals_ids[0][1]:
                    intervals.append(merged_intervals_ids[0])

            if intervals and intervals[-1][1] != len(reference_path):
                intervals.append([intervals[-1][1], len(reference_path)])

            resampled_path = None
            for i in intervals:
                if i in merged_intervals_ids:
                    step = 3.0
                else:
                    step = vehicle_settings["resampling_reference_path"]
                if resampled_path is None:
                    resampled_path = resample_polyline(
                        reference_path[i[0] : i[1]], step
                    )
                else:
                    resampled_path = np.concatenate(
                        (
                            resampled_path,
                            resample_polyline(reference_path[i[0] : i[1]], step),
                        )
                    )
        else:
            resampled_path = reference_path

    else:
        resampled_path = reference_path
    return resampled_path, lanelets_leading_to_goal


def create_lanelet_network(
    lanelet_network: LaneletNetwork, lanelets_leading_to_goal: List[int]
) -> LaneletNetwork:
    """
    Create a new lanelet network based on the current structure and given reference lanelets.
    """
    new_lanelet_network = LaneletNetwork()

    for lanelet_id in lanelets_leading_to_goal:
        lanelet_orig = lanelet_network.find_lanelet_by_id(lanelet_id)

        predecessor = list(
            set(lanelet_orig.predecessor).intersection(lanelets_leading_to_goal)
        )
        successor = list(
            set(lanelet_orig.successor).intersection(lanelets_leading_to_goal)
        )

        lanelet = Lanelet(
            lanelet_orig.left_vertices,
            lanelet_orig.center_vertices,
            lanelet_orig.right_vertices,
            lanelet_orig.lanelet_id,
            predecessor,
            successor,
        )

        if {lanelet_orig.adj_left}.intersection(lanelets_leading_to_goal):
            lanelet.adj_left = lanelet_orig.adj_left
            lanelet.adj_left_same_direction = lanelet_orig.adj_left_same_direction
        if {lanelet_orig.adj_right}.intersection(lanelets_leading_to_goal):
            lanelet.adj_right = lanelet_orig.adj_right
            lanelet.adj_right_same_direction = lanelet_orig.adj_right_same_direction
        new_lanelet_network.add_lanelet(lanelet)
    return new_lanelet_network


def compute_approximating_circle_radius(
    ego_length, ego_width
) -> Tuple[Union[float, Any], Any]:
    """
    From Julia Kabalar
    Computes parameters of the circle approximation of the ego_vehicle

    :param ego_length: Length of ego vehicle
    :param ego_width: Width of ego vehicle
    :return: radius of circle approximation, circle center point distance
    """
    assert ego_length >= 0 and ego_width >= 0, "Invalid vehicle dimensions = {}".format(
        [ego_length, ego_width]
    )

    if np.isclose(ego_length, 0.0) and np.isclose(ego_width, 0.0):
        return 0.0, 0.0

    # Divide rectangle into 3 smaller rectangles
    square_length = ego_length / 3

    # Calculate minimum radius
    diagonal_square = np.sqrt((square_length / 2) ** 2 + (ego_width / 2) ** 2)

    # Round up value
    if diagonal_square > round(diagonal_square, 1):
        approx_radius = round(diagonal_square, 1) + 0.1
    else:
        approx_radius = round(diagonal_square, 1)

    return approx_radius, round(square_length * 2, 1)


def convert_pos_curvilinear(
    state: Union[InitialState, PMState, KSState, CustomState],
    configuration: PlanningConfigurationVehicle,
) -> [float, float]:
    """
    Converts the position of the state to the CLCS.
    """
    if configuration.reference_point == ReferencePoint.REAR:
        pos = configuration.CLCS.convert_to_curvilinear_coords(
            state.position[0] - configuration.wb_ra * np.cos(state.orientation),
            state.position[1] - configuration.wb_ra * np.sin(state.orientation),
        )
    elif configuration.reference_point == ReferencePoint.CENTER:
        pos = configuration.CLCS.convert_to_curvilinear_coords(
            state.position[0], state.position[1]
        )
    else:
        raise ValueError(
            "<compute_initial_state>: unknown reference point: {}".format(
                configuration.reference_point
            )
        )
    return pos


def compute_initial_state(
    initial_state: State, configuration: PlanningConfigurationVehicle
) -> TrajPoint:
    """
    This function computes the initial state of the ego vehicle for the qp-planner given a
    planning problem according to CommonRoad. It is assumed that d*kappa_ref << 1 holds, where d is the distance of
    the ego vehicle to the reference path and kappa_ref is the curvature of the reference path,
    for the transformation of the ego vehicle's velocity to the curvilinear coordinate system.

    :param initial_state: initial state of the planning problem
    :param configuration: parameters of the ego vehicle
    :return: initial state of the ego vehicle in curvilinear coordinates (TrajPoint)
    """
    pos = convert_pos_curvilinear(initial_state, configuration)
    ref_path = np.array(configuration.CLCS.reference_path())
    ref_orientation = compute_orientation_from_polyline(ref_path)
    ref_path_length = compute_pathlength_from_polyline(ref_path)
    orientation_interpolated = np.interp(pos[0], ref_path_length, ref_orientation)

    v_x = initial_state.velocity * np.cos(
        initial_state.orientation - orientation_interpolated
    )

    if hasattr(initial_state, "acceleration"):
        a = initial_state.acceleration
    else:
        a = 0.0

    if hasattr(initial_state, "jerk"):
        j = initial_state.jerk
    else:
        j = 0.0

    # compute orientation in curvilinear coordinate frame (need to be validated in [-pi, pi])
    theta_cl = validate_orientation(
        initial_state.orientation
    )  # - orientation_interpolated)

    # compute curvature
    if hasattr(initial_state, "steering_angle"):
        kr = np.tan(initial_state.steering_angle) / configuration.wheelbase
    elif hasattr(initial_state, "slip_angle"):
        kr = initial_state.slip_angle
    else:
        kr = np.interp(pos[0], configuration.ref_pos, configuration.ref_curv)

    if hasattr(initial_state, "steering_angle_speed"):
        kr_d = (
            initial_state.steering_angle_speed
            * (1 + (kr * configuration.wheelbase) ** 2)
            / configuration.wheelbase
        )
    elif hasattr(initial_state, "yaw_rate"):
        kr_d = initial_state.yaw_rate
    else:
        kr_d = np.interp(pos[0], configuration.ref_pos, configuration.ref_curv)

    return TrajPoint(
        t=0.0,
        x=pos[0],
        v=v_x,
        a=a,
        j=j,
        y=pos[1],
        theta=theta_cl,
        kappa=kr,
        kappa_dot=kr_d,
    )
