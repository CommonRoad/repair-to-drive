from typing import Dict, Tuple, List, Union, Any, Optional
import numpy as np
import os


# commonroad-io
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet

from crrepairer.miqp_planner.configuration import (
    ConfigurationBuilder,
    ReferencePoint,
)

from crrepairer.miqp_planner.initialization import (
    find_reference_path_and_lanelets_leading_to_goal,
)

from crmonitor.common.world import DynamicObstacleVehicle, Vehicle
from commonroad_route_planner.route_planner import RoutePlanner

from crrepairer.smt.monitor_wrapper import ScenarioType
from crrepairer.utils.general import create_curvilinear_coordinate_system


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
    assert (
        planning_problem.planning_problem_id in settings
    ), "Cannot find settings for planning problem {}".format(
        planning_problem.planning_problem_id
    )

    vehicle_settings = settings[planning_problem.planning_problem_id]
    # TODO: create new function instead of using qp planner
    config_builder = ConfigurationBuilder()

    if path_root_configs:
        config_builder.set_root_path(
            root=path_root_configs, path_to_config=path_to_config
        )
    else:
        config_builder.set_root_path(
            root=os.path.normpath(os.path.dirname(__file__)),
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
