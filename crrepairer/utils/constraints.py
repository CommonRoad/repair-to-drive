from typing import Tuple, List, Dict, Union
import numpy as np

# commonroad
from commonroad.common.util import Interval

# commonroad-dc
from commonroad_dc import pycrcc

# commonroad-qp-planner
from crrepairer.miqp_planner.configuration import (
    PlanningConfigurationVehicle,
    ReferencePoint,
)

# commonroad-reach
from commonroad_reach import pycrreach
from commonroad_reach.utility.reach_operation import (
    lon_interval_connected_set,
    lat_interval_connected_set,
    lon_velocity_interval_connected_set,
)

from commonroad_reach_semantic.data_structure.driving_corridor_extractor import (
    DrivingCorridor,
)
from commonroad_reach.data_structure.reach.reach_polygon import ReachPolygon


def longitudinal_position_constraints(
    longitudinal_driving_corridor: DrivingCorridor, FULL=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes longitudinal position constraints for the reference point from a given longitudinal driving corridor.
    See Manzinger et al. (2020): "Using reachable sets for trajectory planning of automated vehicles", Sec. VI-A

    :param longitudinal_driving_corridor:
    :return: Tuple (s_min, s_max) with arrays for lon position constraints
    """
    longitudinal_constraints = list()
    if hasattr(longitudinal_driving_corridor, "list_nodes_kripke"):
        for connected_set in longitudinal_driving_corridor.list_nodes_kripke:
            longitudinal_constraints.append(
                lon_interval_connected_set(list(connected_set.set_nodes_reach))
            )
    else:
        for _, connected_set in longitudinal_driving_corridor.dict_step_to_cc.items():
            longitudinal_constraints.append(
                lon_interval_connected_set(list(connected_set.list_nodes_reach))
            )

    longitudinal_constraints = np.array(longitudinal_constraints)
    if FULL:
        s_min = longitudinal_constraints[:, 0]
        s_max = longitudinal_constraints[:, 1]
    else:
        s_min = longitudinal_constraints[1:, 0]
        s_max = longitudinal_constraints[1:, 1]
    return s_min, s_max


def longitudinal_velocity_constraints(
    longitudinal_driving_corridor: DrivingCorridor, FULL=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes longitudinal velocity constraints for the reference point from a given longitudinal driving corridor.

    :param longitudinal_driving_corridor:
    :return: Tuple (v_min, v_max) with arrays for lon position constraints
    """
    longitudinal_constraints = list()
    if hasattr(longitudinal_driving_corridor, "list_nodes_kripke"):
        for connected_set in longitudinal_driving_corridor.list_nodes_kripke:
            longitudinal_constraints.append(
                lon_velocity_interval_connected_set(list(connected_set.set_nodes_reach))
            )
    else:
        for _, connected_set in longitudinal_driving_corridor.dict_step_to_cc.items():
            longitudinal_constraints.append(
                lon_velocity_interval_connected_set(
                    list(connected_set.list_nodes_reach)
                )
            )
    longitudinal_constraints = np.array(longitudinal_constraints)
    if FULL:
        v_min = longitudinal_constraints[:, 0]
        v_max = longitudinal_constraints[:, 1]
    else:
        v_min = longitudinal_constraints[1:, 0]
        v_max = longitudinal_constraints[1:, 1]
    return v_min, v_max


def lateral_position_constraints(
    lateral_driving_corridor: DrivingCorridor,
    longitudinal_driving_corridor: DrivingCorridor,
    longitudinal_positions: List[float],
    vehicle_configuration: PlanningConfigurationVehicle,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates the lateral position constraints from a given lateral driving corridor.

    :return: Tuple (d_min, d_max) with lateral constraints for the three circles
    """
    if hasattr(longitudinal_driving_corridor, "list_nodes_kripke"):
        long_node_dict = longitudinal_driving_corridor.list_nodes_kripke
        length_long = len(long_node_dict)
    else:
        long_node_dict = longitudinal_driving_corridor.dict_step_to_cc
        length_long = len(long_node_dict.keys())
    if hasattr(lateral_driving_corridor, "list_nodes_kripke"):
        lat_node_dict = lateral_driving_corridor.list_nodes_kripke
        length_lat = len(lat_node_dict)
    else:
        lat_node_dict = lateral_driving_corridor.dict_step_to_cc
        length_lat = len(lat_node_dict.keys())

    assert (
        length_long == length_lat
    ), "longitudinal and lateral driving corridor have to have the same number of time steps"

    # get time steps from driving corridor
    time_steps = list(longitudinal_driving_corridor.keys())

    # Dict with longitudinal positions over time
    lon_positions_dict = dict(zip(time_steps, longitudinal_positions))

    # lateral constraints for reference point circle (rear or center)
    d_constraints_reference_point = _lateral_position_constraints_reference_point(
        lateral_driving_corridor, time_steps
    )

    # lateral constraints for other circles (rear/center and front)
    d_constraints_rear_or_center, d_constraints_front = (
        _lateral_position_constraints_other_circles(
            longitudinal_driving_corridor,
            lon_positions_dict,
            vehicle_configuration,
            d_constraints_reference_point,
        )
    )

    # get d_min and d_max constraints for all three circle
    d_reference = np.array(list(d_constraints_reference_point.values()))
    d_min_reference, d_max_reference = d_reference[:, 0], d_reference[:, 1]
    d_min_front, d_max_front = d_constraints_front
    d_min_rear_or_center, d_max_rear_or_center = d_constraints_rear_or_center

    # create d_min/d_max arrays with lateral constraints for all three circles
    d_min = np.array(
        (d_min_reference[1:], d_min_rear_or_center[1:], d_min_front[1:])
    ).transpose()
    d_max = np.array(
        (d_max_reference[1:], d_max_rear_or_center[1:], d_max_front[1:])
    ).transpose()

    return d_min, d_max


def _lateral_position_constraints_reference_point(
    lateral_driving_corridor: DrivingCorridor, time_steps: List[int]
) -> Dict[int, np.array]:
    """
    Lateral position constraints for the center of the reference circle are directly obtained from the
    boundary of the lateral driving corridor.
    See Manzinger et al. (2020): "Using reachable sets for trajectory planning of automated vehicles", Sec. VI-B1)
    """
    d_constraints_reference_point = dict()
    for time_idx in time_steps:
        # connected set at time step
        if hasattr(lateral_driving_corridor, "dict_step_to_cc"):
            connected_set = lateral_driving_corridor.dict_step_to_cc[time_idx]
            d_constraints_reference_point[time_idx] = np.array(
                lat_interval_connected_set(connected_set.list_nodes_reach)
            )

        else:
            connected_set = lateral_driving_corridor.list_nodes_kripke[time_idx]
            d_constraints_reference_point[time_idx] = np.array(
                lat_interval_connected_set(connected_set.set_nodes_reach)
            )

    return d_constraints_reference_point


def _lateral_position_constraints_other_circles(
    longitudinal_driving_corridor: DrivingCorridor,
    longitudinal_positions: Dict[int, float],
    vehicle_configuration: PlanningConfigurationVehicle,
    d_constraints_reference_point: Dict[int, np.array],
) -> Tuple[Tuple, Tuple]:
    """
    Computes the lateral position constraints for the rear/center and front circles of the shape of the ego vehicle.
    See Manzinger et al. (2020): "Using reachable sets for trajectory planning of automated vehicles", Sec. VI-B2)
    """
    # initialize lists for rear/center and front
    d_constraints_rear_or_center = list()
    d_constraints_front = list()
    for time_step in longitudinal_positions.keys():
        # connected set from DC at time step
        if hasattr(longitudinal_driving_corridor, "list_nodes_kripke"):
            connected_set = longitudinal_driving_corridor.list_nodes_kripke[time_step]
        else:
            connected_set = longitudinal_driving_corridor.dict_step_to_cc[time_step]
        # current longitudinal position of reference point
        s = longitudinal_positions[time_step]

        # normal and tangent vectors at s
        normal = vehicle_configuration.CLCS.normal(s)
        tangent = vehicle_configuration.CLCS.tangent(s)

        # Cartesian position coordinates of reference point
        cartesian_position = vehicle_configuration.CLCS.convert_to_cartesian_coords(
            s, 0.0
        )

        # find longitudinal coordinates of other circles in cartesian coordinates
        if vehicle_configuration.reference_point == ReferencePoint.CENTER:
            cartesian_position_rear_or_center = (
                cartesian_position - vehicle_configuration.wb_ra * tangent
            )
            cartesian_position_front = (
                cartesian_position + vehicle_configuration.wb_fa * tangent
            )
        elif vehicle_configuration.reference_point == ReferencePoint.REAR:
            cartesian_position_rear_or_center = (
                cartesian_position + vehicle_configuration.wb_ra * tangent
            )
            cartesian_position_front = (
                cartesian_position
                + (vehicle_configuration.wb_ra + vehicle_configuration.wb_fa) * tangent
            )
        else:
            raise ValueError(
                "<_lateral_position_constraints_other_circles>: reference point of the ego vehicle is"
                " unknown: {}".format(vehicle_configuration.reference_point)
            )

        # determine drivable area and create collision polygon
        drivable_area = list()
        if hasattr(connected_set, "set_nodes_reach"):
            nodes = list(connected_set.set_nodes_reach)
        else:
            nodes = list(connected_set.list_nodes_reach)

        for node in nodes:

            if type(node) == pycrreach.ReachNode:
                # C++ backend
                position_rect = node.position_rectangle
            else:
                # Python backend
                position_rect = node.position_rectangle
                # create cartesian collision polygon
            poly_co = _position_rectangle_to_cartesian_collision_polygon(
                position_rect, vehicle_configuration.CLCS
            )
            # add polygon to drivable area list
            drivable_area.append(poly_co)

        # create collision checker object for raytracing
        cc = pycrcc.CollisionChecker()
        [cc.add_collision_object(poly) for poly in drivable_area]

        # determine lateral constraints for rear or center circle
        d_constraints_rear_or_center.append(
            _find_admissible_intervals(
                cartesian_position_rear_or_center,
                normal,
                cc,
                (
                    d_constraints_reference_point[time_step][0],
                    d_constraints_reference_point[time_step][1],
                ),
            )
        )

        # determine lateral constraints for front circle
        d_constraints_front.append(
            _find_admissible_intervals(
                cartesian_position_front,
                normal,
                cc,
                (
                    d_constraints_reference_point[time_step][0],
                    d_constraints_reference_point[time_step][1],
                ),
            )
        )
    d_constraints_rear_or_center = np.array(d_constraints_rear_or_center)
    d_constraints_front = np.array(d_constraints_front)

    return (d_constraints_rear_or_center[:, 0], d_constraints_rear_or_center[:, 1]), (
        d_constraints_front[:, 0],
        d_constraints_front[:, 1],
    )


def _find_admissible_intervals(
    position: np.ndarray,
    normal_vector: np.ndarray,
    collision_checker: pycrcc.CollisionChecker,
    d_reference: Tuple,
):
    point_1 = position - 500.0 * normal_vector
    point_2 = position + 500.0 * normal_vector

    d_reference_interval = Interval(d_reference[0], d_reference[1])
    interval_list = collision_checker.raytrace(
        point_1[0], point_1[1], point_2[0], point_2[1], False
    )
    if len(interval_list) > 0:
        interval_list = collision_checker.raytrace(
            point_1[0], point_1[1], point_2[0], point_2[1], True
        )
    admissible_intervals = list()
    # get d-coordinates along straight line spanned by position and normal_vector
    for i in interval_list:
        d = list()
        # compute signed distance between interval start/end point and position
        d.append(np.dot(normal_vector, np.array([i[0], i[1]]) - position))
        d.append(np.dot(normal_vector, np.array([i[2], i[3]]) - position))
        interval = Interval(min(d), max(d))
        if interval.overlaps(d_reference_interval):
            admissible_intervals.append(interval)
    admissible_intervals.append(d_reference_interval)
    merged_interval = _merge_connected_intervals(admissible_intervals)
    return float(merged_interval.start), float(merged_interval.end)


def _merge_connected_intervals(interval_list: List[Interval]) -> Interval:
    """
    Merges a list of connected intervals.
    """
    interval_list.sort(key=lambda interval: interval.start)
    start = interval_list[0].start
    interval_list.sort(key=lambda interval: interval.end, reverse=True)
    end = interval_list[0].end
    return Interval(start, end)


# TODO Adapt to new DC version
def _position_rectangle_to_cartesian_collision_polygon(
    position_rect: Union[ReachPolygon, pycrreach.ReachPolygon], ccosy
):
    """
    Converts a curvilinear position reactangle (drivable area) to Cartesian frame and creates a collision polygon
    """
    # convert curvilinear position rectangle to cartesian polygon
    if type(position_rect) == pycrreach.ReachPolygon:
        poly_cart_vertices, triangle_mesh = ccosy.convert_rectangle_to_cartesian_coords(
            position_rect.p_lon_min,
            position_rect.p_lon_max,
            position_rect.p_lat_min,
            position_rect.p_lat_max,
        )
    else:
        poly_cart_vertices, triangle_mesh = ccosy.convert_rectangle_to_cartesian_coords(
            position_rect.p_lon_min,
            position_rect.p_lon_max,
            position_rect.p_lat_min,
            position_rect.p_lat_max,
        )

    # create list of pycrcc.Triangles
    triangle_mesh_co = [
        pycrcc.Triangle(
            tri[0][0], tri[0][1], tri[1][0], tri[1][1], tri[2][0], tri[2][1]
        )
        for tri in triangle_mesh
    ]

    # create pycrcc.Polygon for drivable area
    poly_co = pycrcc.Polygon(poly_cart_vertices, list(), triangle_mesh_co)
    return poly_co
