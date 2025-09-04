from typing import Tuple
import logging
import os.path
import numpy as np
from matplotlib import pyplot as plt

from commonroad.common.file_reader import CommonRoadFileReader

from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import (
    DynamicObstacleParams,
    TrajectoryParams,
    OccupancyParams,
)

from crrepairer.miqp_planner.trajectory import Trajectory

LOGGER = logging.getLogger(__name__)


def open_scenario(settings):
    scenario_path = os.path.abspath(os.path.join(__file__))
    crfr = CommonRoadFileReader(
        scenario_path + settings["scenario_settings"]["scenario_name"] + ".xml"
    )
    scenario, planning_problem_set = crfr.open()
    return scenario, planning_problem_set


def plot_result(scenario, ego_vehicle, ax=None):

    rnd = MPRenderer(figsize=(20, 10), ax=ax)

    scenario.draw(rnd)

    draw_params = OccupancyParams()
    draw_params.shape.opacity = 0.1
    for obstacle in scenario.dynamic_obstacles:
        for o in obstacle.prediction.occupancy_set:
            o.draw(rnd, draw_params=draw_params)

    # plot ego trajectory
    draw_params = DynamicObstacleParams()
    draw_params.vehicle_shape.occupancy.shape.facecolor = "black"
    draw_params.vehicle_shape.occupancy.shape.edgecolor = "black"
    ego_vehicle.draw(rnd, draw_params=draw_params)

    draw_params = OccupancyParams()
    draw_params.shape.opacity = 0.1
    draw_params.shape.facecolor = "black"
    for o in ego_vehicle.prediction.occupancy_set:
        o.draw(rnd, draw_params=draw_params)
    rnd.render()


def plot_position_constraints(
    trajectory_cvln: Trajectory,
    s_constraints: Tuple[np.ndarray, np.ndarray],
    d_constraints: Tuple[np.ndarray, np.ndarray],
):
    s_min, s_max = s_constraints
    d_min, d_max = d_constraints

    fig, axs = plt.subplots(2)

    # s_limit
    axs[0].plot(list(range(len(s_min))), s_min, color="red", label="bounds")
    axs[0].plot(list(range(len(s_max))), s_max, color="red")
    axs[0].plot(
        list(range(len(trajectory_cvln.states) - 1)),
        [state.position[0] for state in trajectory_cvln.states[1:]],
        color="black",
        label="planned",
    )
    axs[0].set_xlabel("time step")
    axs[0].set_ylabel("s")
    axs[0].legend()

    # d_limit
    axs[1].plot(list(range(len(d_min))), d_min, color="red")
    axs[1].plot(list(range(len(d_max))), d_max, color="red")
    axs[1].plot(
        list(range(len(trajectory_cvln.states) - 1)),
        [state.position[1] for state in trajectory_cvln.states[1:]],
        color="black",
    )
    axs[1].set_xlabel("time step")
    axs[1].set_ylabel("d")

    plt.tight_layout()
    plt.show()


def validate_orientation(angle: float) -> float:
    while angle > np.pi:
        angle = angle - 2 * np.pi
    while angle < -np.pi:
        angle = angle + 2 * np.pi
    return angle


def calculate_safe_distance(v_follow, v_lead, a_min_lead, a_min_follow, t_react_follow):
    d_safe = (
        (v_lead**2) / (-2 * np.abs(a_min_lead))
        - (v_follow**2) / (-2 * np.abs(a_min_follow))
        + v_follow * t_react_follow
    )

    return d_safe


def derivative_safe_distance(ego_v, ego_a, t_react):
    return ego_v / np.abs(ego_a) + t_react
