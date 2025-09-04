# standard imports
import copy
from enum import Enum
from typing import List, Union, Optional
from shapely.geometry.polygon import Polygon

# third party
import matplotlib.pyplot as plt
import numpy as np

# commonroad-io
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.visualization.draw_params import (
    MPDrawParams,
    DynamicObstacleParams,
    ShapeParams,
    OccupancyParams,
    PlanningProblemParams,
)
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.renderer import IRenderer
from commonroad.geometry.shape import Rectangle, Circle
from crrepairer.miqp_planner.utils import calculate_safe_distance

from crmonitor.common.world import World

from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer


class TUMColor(List, Enum):
    TUMblue = [0, 101 / 255, 189 / 255]
    TUMgreen = [162 / 255, 173 / 255, 0]
    TUMgray = [156 / 255, 157 / 255, 159 / 255]
    TUMdarkgray = [88 / 255, 88 / 255, 99 / 255]
    TUMorange = [227 / 255, 114 / 255, 34 / 255]
    TUMdarkblue = [0, 82 / 255, 147 / 255]
    TUMwhite = [1, 1, 1]
    TUMblack = [0, 0, 0]
    TUMyellow = [203 / 255, 171 / 255, 1 / 255]
    TUMlightBlue = [100 / 255, 160 / 255, 200 / 255]
    TUMlightgray = [217 / 255, 218 / 255, 219 / 255]


def draw_scenario(
    ax,
    scenario,
    planning_problem,
    step,
    ego_vehicle,
    tv,
    tc,
    draw_occupancies_ego=True,
    draw_planning_problem=True,
    draw_occupancies_other=False,
    y_shift=0.0,
    title="",
    flag_violation=False,
):
    rnd = MPRenderer(ax=ax)

    rnd.plot_limits = [20.0, 120.0, -2.0 + y_shift, 5.0 + y_shift]

    params = MPDrawParams()
    params.time_begin = step
    params.dynamic_obstacle.draw_icon = True

    scenario.draw(rnd, draw_params=params)

    try:
        other_obs = copy.deepcopy(scenario.obstacle_by_id(102))
        for state in other_obs.prediction.trajectory.state_list:
            state.position[0] -= 2
        params.dynamic_obstacle.occupancy.shape.opacity = 0.2
        other_obs.draw(rnd, draw_params=params)
    except KeyError:
        pass

    if draw_planning_problem:
        pp_params = PlanningProblemParams()
        pp_params.initial_state.state.draw_arrow = False
        pp_params.initial_state.state.radius = 0.4
        planning_problem.draw(rnd, draw_params=pp_params)

    ego_params = DynamicObstacleParams()
    ego_params.time_begin = step
    ego_params.trajectory.draw_trajectory = False
    ego_params.vehicle_shape.occupancy.shape.facecolor = "#e37222"
    ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9c4100"
    ego_params.draw_icon = True
    ego_params.vehicle_shape.direction.zorder = 55
    ego_params.vehicle_shape.occupancy.shape.zorder = 55
    ego_params.zorder = 55

    if ego_vehicle is not None:
        ego_vehicle.draw(rnd, draw_params=ego_params)

        if draw_occupancies_ego:
            occ_params = OccupancyParams()
            occ_params.shape.facecolor = "#E37222"
            occ_params.shape.edgecolor = "#9C4100"
            occ_params.shape.opacity = 0.15
            occ_params.shape.zorder = 50
            for i, occ in enumerate(ego_vehicle.prediction.occupancy_set):
                if flag_violation:
                    if i >= tv:
                        occ_params.shape.facecolor = "red"
                        occ_params.shape.edgecolor = "red"
                else:
                    if i >= tc:
                        occ_params.shape.facecolor = "#a2ad00"
                        occ_params.shape.edgecolor = "#a2ad00"
                occ.draw(rnd, draw_params=occ_params)

        pos = np.array(
            [s.position for s in ego_vehicle.prediction.trajectory.state_list]
        )
        pos[:, 1] += y_shift
        rnd.ax.plot(pos[:, 0], pos[:, 1], color="#9C4100", linewidth=3.0, zorder=21)

    if draw_occupancies_other:
        occ_params = OccupancyParams()
        occ_params.shape.opacity = 0.05
        for obs in scenario.dynamic_obstacles:
            for o in obs.prediction.occupancy_set:
                o.draw(rnd, draw_params=occ_params)

    for pos in [
        [99, 4],
        [103, 3.166],
        [107, 2.333],
        [111, 1.5],
        [115, 1.5],
        [119, 1.5],
    ]:
        draw_traffic_cone(rnd, np.array([pos[0], pos[1] + y_shift]))

    rnd.ax.set_title(title, fontsize=16)
    rnd.render()


def draw_traffic_cone(rnd: IRenderer, position: np.ndarray):
    for r, color in zip([0.4, 0.3, 0.2], ["#fc6716", "#ffffff", "#fc6716"]):
        Circle(r, position).draw(
            rnd,
            draw_params=ShapeParams(
                facecolor=color, edgecolor=color, zorder=1000 + int(r * 10)
            ),
        )


def visualize_repaired_result(
    config: RepairerConfiguration,
    ego_initial: DynamicObstacle,
    ego_repaired: DynamicObstacle,
    repairer: SMTTrajectoryRepairer,
    plot_velocity: bool = True,
    plot_acceleration: bool = True,
):
    if plot_velocity:
        visualize_v_profile(
            ego_initial,
            ego_repaired,
            config.repair.t_0,
            config.repair.t_f,
            repairer.tc,
            repairer.tv,
        )

    if plot_acceleration:
        visualize_a_profile(
            config.scenario.dt,
            ego_initial,
            ego_repaired,
            config.repair.t_0,
            config.repair.t_f,
            repairer.tc,
            repairer.tv,
        )
    for time_step in range(config.repair.t_0, config.repair.t_f):
        if config.debug.save_plots:
            path_fig = config.general.path_figures
        else:
            path_fig = None
        visualize_scenario(
            config.scenario,
            ego_initial,
            ego_repaired,
            time_step,  # Assuming time_end is the current time_step for visualization
            path_fig,
            config.debug.plot_limits,
            config.repair.t_f,
            repairer.tc,
            repairer.tv,
            repairer.target_vehicle,
        )


def plot_velocity_acceleration_profile(
    time_list: List[int],
    velocity_list: List[float],
    color: Union[List, str],
    label: str,
    linestyle: str = "-",
    marker: str = ".",
):
    plt.plot(
        time_list,
        velocity_list,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=1.5,
        label=label,
    )


def visualize_v_profile_tc_all(
    repairer: SMTTrajectoryRepairer,
    ego_initial: DynamicObstacle,
    ego_repaired: DynamicObstacle,
    time_start: int,
    time_end: int,
    ylim: Optional[List[float]] = None,
    figsize=(6, 2.4),
    velocity_limit=None,
):
    plt.figure(figsize=figsize)
    tv = repairer.tv
    tc = repairer.tc
    time_list = [time_step - time_start for time_step in range(time_start, time_end)]
    ego_ini_vel = [
        ego_initial.state_at_time(t).velocity for t in range(time_start, time_end)
    ]
    ego_rep_vel = [
        ego_repaired.state_at_time(t).velocity for t in range(time_start, time_end)
    ]
    if velocity_limit:
        plt.axhline(y=velocity_limit, linestyle="--", linewidth=1.0)
    plot_velocity_acceleration_profile(
        time_list, ego_ini_vel, TUMColor.TUMblue.value, "Initial", marker="x"
    )
    plot_velocity_acceleration_profile(
        time_list[tv - time_start :],
        ego_ini_vel[tv - time_start :],
        "red",
        "Violation",
        marker="x",
    )
    plot_velocity_acceleration_profile(
        time_list[tc - time_start :],
        ego_rep_vel[tc - time_start :],
        TUMColor.TUMgreen.value,
        "Repaired",
    )

    for state_list in repairer.t_solver.tc_object.state_list_set:
        plt.plot(
            [state.time_step - time_start for state in state_list],
            [state.velocity for state in state_list],
            color=TUMColor.TUMgray.value,
            linestyle=":",
            linewidth=1.5,
        )
    plt.xticks(range(0, time_end - time_start, 10))
    if ylim:
        plt.ylim(ylim)
    plt.xlim([0, time_end - time_start])
    plt.xlabel("Time step")
    plt.ylabel("Velocity")
    plt.show()


def visualize_v_profile(
    ego_initial: DynamicObstacle,
    ego_repaired: DynamicObstacle,
    time_start: int,
    time_end: int,
    tc: int,
    tv: int,
):
    plt.figure(figsize=(6, 2.4))
    time_list = [time_step - time_start for time_step in range(time_start, time_end)]
    ego_ini_vel = [
        ego_initial.state_at_time(t).velocity for t in range(time_start, time_end)
    ]
    ego_rep_vel = [
        ego_repaired.state_at_time(t).velocity for t in range(time_start, time_end)
    ]

    plt.axhline(y=0, linestyle="--", linewidth=1.0)
    plot_velocity_acceleration_profile(
        time_list, ego_ini_vel, TUMColor.TUMblue.value, "Initial"
    )
    plot_velocity_acceleration_profile(
        time_list[tv - time_start :], ego_ini_vel[tv - time_start :], "red", "Violation"
    )
    plot_velocity_acceleration_profile(
        time_list[tc - time_start :],
        ego_rep_vel[tc - time_start :],
        TUMColor.TUMgreen.value,
        "Repaired",
    )

    plt.xticks(range(0, time_end - time_start, 10))
    plt.xlim([0, time_end - time_start])
    plt.xlabel("Time step")
    plt.ylabel("Velocity")
    plt.show()


def calculate_acceleration(
    obstacle: DynamicObstacle, time_step: int, dt: float
) -> float:
    if hasattr(obstacle.state_at_time(time_step), "acceleration"):
        return obstacle.state_at_time(time_step).acceleration
    else:
        return (
            obstacle.state_at_time(time_step + 1).velocity
            - obstacle.state_at_time(time_step).velocity
        ) / dt


def visualize_a_profile(
    dt: float,
    ego_initial: DynamicObstacle,
    ego_repaired: DynamicObstacle,
    time_start: int,
    time_end: int,
    tc: int,
    tv: int,
):
    time_list = [t - time_start for t in range(time_start, time_end)]
    ego_ini_acc = [
        calculate_acceleration(ego_initial, t, dt) for t in range(time_start, time_end)
    ]
    ego_rep_acc = [
        calculate_acceleration(ego_repaired, t, dt) for t in range(time_start, time_end)
    ]

    plt.figure(figsize=(6, 2.4))
    plot_velocity_acceleration_profile(
        time_list, ego_ini_acc, TUMColor.TUMblue.value, "Initial"
    )
    plot_velocity_acceleration_profile(
        time_list[tv - time_start :], ego_ini_acc[tv - time_start :], "red", "Violation"
    )
    plot_velocity_acceleration_profile(
        time_list[tc - time_start :],
        ego_rep_acc[tc - time_start :],
        TUMColor.TUMgreen.value,
        "Repaired",
    )

    plt.xticks(range(0, time_end - time_start, 5))
    plt.xlabel("Time step")
    plt.ylabel("Acceleration")
    plt.show()


def visualize_scenario(
    scenario: Scenario,
    ego_initial: DynamicObstacle,
    ego_repaired: DynamicObstacle,
    time_step: int,
    save_path: str = None,
    plot_limits=None,
    end_time=None,
    tc=None,
    tv=None,
    target_veh=None,
):
    """
    Function to visualize the repairing result given time step
    :param scenario: CommonRoad scenario object
    :param ego_initial: initially-planned trajectory
    :param ego_repaired: repaired ego vehicle
    :param time_step: current time step
    :param save_path: Path to save plot as .png/.svg (optional)
    :param plot_limits: plot limits of the scenario
    :param end_time: ending time step
    :param tc: time-to-comply
    :param tv: time-to-violation
    :param target_veh: target vehicle for repairing
    :param world: world state
    """
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 10))
    rnd_0 = MPRenderer(ax=ax0, plot_limits=plot_limits)
    rnd_1 = MPRenderer(ax=ax1, plot_limits=plot_limits)

    # visualize scenario
    for rnd in (rnd_0, rnd_1):
        rnd.draw_params.time_begin = time_step
        if end_time:
            rnd.draw_params.time_end = end_time
        rnd.draw_params.trajectory.draw_trajectory = False
        rnd.draw_params.lanelet_network.lanelet.fill_lanelet = False
        rnd.draw_params.occupancy.draw_occupancies = False
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = (
            False
        )
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
            TUMColor.TUMblack.value
        )
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = (
            TUMColor.TUMblack.value
        )
        rnd.draw_params.dynamic_obstacle.draw_shape = True
        rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = True
        rnd.draw_params.dynamic_obstacle.draw_signals = False
        rnd.draw_params.dynamic_obstacle.draw_icon = True
        # rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = True
        # rnd.draw_params.traffic_sign.draw_traffic_signs = True
        rnd.draw_params.lanelet_network.lanelet.stop_line_color = (
            TUMColor.TUMblack.value
        )
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
            TUMColor.TUMgray.value
        )
        rnd.draw_params.lanelet_network.lanelet.draw_stop_line = True
        scenario.draw(rnd)

    rnd_0.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = True
    rnd_1.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = True
    rnd_0.draw_params.dynamic_obstacle.draw_shape = True
    rnd_0.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False
    rnd_1.draw_params.dynamic_obstacle.draw_shape = True
    rnd_1.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False

    if time_step >= tv:
        ego_color = "red"
    else:
        ego_color = TUMColor.TUMblue.value
    ego_mark = "x"

    rnd_0.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.opacity = 0.5
    rnd_0.draw_params.dynamic_obstacle.occupancy.draw_occupancies = True
    rnd_0.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
        TUMColor.TUMblue.value
    )
    ego_initial.draw(rnd_0)

    if target_veh:
        rnd_0.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
            "red"
        )
        rnd_0.draw_params.dynamic_obstacle.occupancy.shape.facecolor = "red"

        target_veh.draw(rnd_0)

    # render scenario and ego vehicle
    rnd_0.render()

    pos_x_initial = [ego_initial.initial_state.position[0]]
    pos_y_initial = [ego_initial.initial_state.position[1]]
    for state in ego_initial.prediction.trajectory.state_list:
        pos_x_initial.append(state.position[0])
        pos_y_initial.append(state.position[1])

    if time_step >= tv:
        rnd_0.ax.plot(
            pos_x_initial[time_step:end_time],
            pos_y_initial[time_step:end_time],
            color=ego_color,
            marker=ego_mark,
            markersize=7.5,
            zorder=35,
            linewidth=1.5,
            label="initial trajectory",
        )
    else:
        rnd_0.ax.plot(
            pos_x_initial[time_step:end_time],
            pos_y_initial[time_step:end_time],
            color=ego_color,
            marker=ego_mark,
            markersize=7.5,
            zorder=35,
            linewidth=1.5,
            label="initial trajectory",
        )

    if time_step >= tc:
        ego_color = TUMColor.TUMgreen.value
        ego_mark = "."
    else:
        ego_color = TUMColor.TUMblue.value
        ego_mark = "x"

    rnd_1.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.opacity = 0.5
    rnd_1.draw_params.dynamic_obstacle.occupancy.draw_occupancies = True

    rnd_1.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
        TUMColor.TUMgreen.value
    )
    ego_repaired.draw(rnd_1)

    if target_veh:
        rnd_1.draw_params.dynamic_obstacle.occupancy.shape.facecolor = "red"
        rnd_1.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
            "red"
        )
        target_veh.draw(rnd_1)

    # render scenario and ego vehicle
    rnd_1.render()

    pos_x_repaired = []
    pos_y_repaired = []
    for state in ego_repaired.prediction.trajectory.state_list:
        pos_x_repaired.append(state.position[0])
        pos_y_repaired.append(state.position[1])

    # visualize optimal trajectory
    rnd_1.ax.plot(
        pos_x_repaired[time_step:end_time],
        pos_y_repaired[time_step:end_time],
        color=ego_color,
        marker=ego_mark,
        markersize=7.5,
        zorder=22,
        linewidth=1.5,
        label="repaired trajectory",
    )

    # if target_veh:
    #     ego_veh_state_ini = ego_initial.state_at_time(time_step)
    #     ego_veh_state_rep = ego_repaired.state_at_time(time_step)
    #     tar_veh_state = target_veh.state_at_time(time_step)
    #     tar_veh_lane = world.vehicle_by_id(target_veh.obstacle_id).get_lane(time_step)
    #     unsafe_poly_ini = compute_unsafe_polygon(
    #         ego_veh_state_ini, tar_veh_state, target_veh, tar_veh_lane
    #     )
    #     rnd_0.ax.fill(
    #         *unsafe_poly_ini.exterior.xy,
    #         zorder=30,
    #         alpha=0.2,
    #         facecolor=TUMcolor.TUMorange.value,
    #         edgecolor=None,
    #     )
    #     unsafe_poly_rep = compute_unsafe_polygon(
    #         ego_veh_state_rep, tar_veh_state, target_veh, tar_veh_lane
    #     )
    #     rnd_1.ax.fill(
    #         *unsafe_poly_rep.exterior.xy,
    #         zorder=30,
    #         alpha=0.2,
    #         facecolor=TUMcolor.TUMorange.value,
    #         edgecolor=None,
    #     )

    ax0.set_title("Initial configuration.")
    ax1.set_title("Repaired configuration.")

    # show plot
    for ax in (ax0, ax1):
        ax.set_xticks([])
        ax.set_yticks([])
        if plot_limits:
            ax.set_xlim([plot_limits[0], plot_limits[1]])
            ax.set_ylim([plot_limits[2], plot_limits[3]])

    # save as .svg file
    if save_path is not None:
        if time_step < 10:
            plt.savefig(
                f"{save_path}/{str(scenario.scenario_id)}_{0}{time_step}.svg",
                format="svg",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"{save_path}/{str(scenario.scenario_id)}_{time_step}.svg",
                format="svg",
                dpi=300,
                bbox_inches="tight",
            )
    else:
        plt.show(block=True)
    # After you're done with a figure
    plt.close(fig)


def visualize_scenario_once(
    scenario: Scenario,
    ego_initial: DynamicObstacle,
    ego_repaired: DynamicObstacle,
    time_step: int,
    save_path: str = None,
    plot_limits=None,
    end_time=None,
    tc=None,
    tv=None,
    target_veh=None,
    world: World = None,
    flag_repair=False,
    marksize=5,
    lanewidth=1.5,
    marker_linewidth=1.5,
):
    """
    Function to visualize the repairing result given time step
    :param scenario: CommonRoad scenario object
    :param ego_initial: initially-planned trajectory
    :param ego_repaired: repaired ego vehicle
    :param time_step: current time step
    :param save_path: Path to save plot as .png/.svg (optional)
    :param plot_limits: plot limits of the scenario
    :param end_time: ending time step
    :param tc: time-to-comply
    :param tv: time-to-violation
    :param target_veh: target vehicle for repairing
    :param world: world state
    :param flag_repair: if True, plot ego_repaired instead of ego_initial
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    rnd = MPRenderer(ax=ax, plot_limits=plot_limits)

    # visualize scenario
    rnd.draw_params.time_begin = time_step
    if end_time:
        rnd.draw_params.time_end = end_time
    rnd.draw_params.trajectory.draw_trajectory = False
    rnd.draw_params.lanelet_network.lanelet.fill_lanelet = False
    rnd.draw_params.occupancy.draw_occupancies = False
    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = False
    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
        TUMColor.TUMgray.value
    )
    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = (
        TUMColor.TUMblack.value
    )
    rnd.draw_params.dynamic_obstacle.draw_shape = True
    rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = True
    rnd.draw_params.dynamic_obstacle.trajectory.line_width = 0.3
    rnd.draw_params.dynamic_obstacle.draw_signals = False
    rnd.draw_params.dynamic_obstacle.draw_icon = True
    # rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = True
    # rnd.draw_params.traffic_sign.draw_traffic_signs = True
    rnd.draw_params.lanelet_network.lanelet.stop_line_color = TUMColor.TUMblack.value
    rnd.draw_params.lanelet_network.lanelet.draw_stop_line = True
    scenario.draw(rnd)

    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = False
    rnd.draw_params.dynamic_obstacle.draw_shape = True
    rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False

    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.opacity = 0.5
    rnd.draw_params.dynamic_obstacle.occupancy.draw_occupancies = False
    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = (
        TUMColor.TUMblue.value
    )

    # Select the correct trajectory based on flag_repair
    ego_to_plot = ego_repaired if flag_repair else ego_initial
    ego_to_plot.draw(rnd)

    # render scenario and ego vehicle
    rnd.render()

    # Extract positions for the selected ego trajectory
    if ego_to_plot.prediction.initial_time_step == ego_to_plot.initial_state.time_step:
        pos_x = []
        pos_y = []
    else:
        pos_x = [ego_to_plot.initial_state.position[0]]
        pos_y = [ego_to_plot.initial_state.position[1]]
    for i in range(
        ego_to_plot.prediction.initial_time_step,
        ego_to_plot.prediction.final_time_step + 1,
    ):
        pos_x.append(ego_to_plot.state_at_time(i).position[0])
        pos_y.append(ego_to_plot.state_at_time(i).position[1])

    if flag_repair:
        if time_step <= tc:
            # Plot the segment from time_step to tc (before TC)
            rnd.ax.plot(
                pos_x[time_step : tc + 1],
                pos_y[time_step : tc + 1],
                color=TUMColor.TUMblue.value,  # Use TUM blue for before TC
                marker="x",
                markersize=marksize,
                zorder=35,
                linewidth=lanewidth,
                markeredgewidth=marker_linewidth,
            )
            # Plot the segment from tc to end_time (after TC)
            rnd.ax.plot(
                pos_x[tc : end_time + 1],
                pos_y[tc : end_time + 1],
                color=TUMColor.TUMgreen.value,  # Use TUM green for after TC
                marker=".",
                markersize=marksize,
                zorder=35,
                markeredgewidth=marker_linewidth,  # Set your desired marker line width here
                linewidth=lanewidth,
            )
        else:
            # If time_step > tc, plot only the segment from time_step to end_time
            rnd.ax.plot(
                pos_x[time_step : end_time + 1],
                pos_y[time_step : end_time + 1],
                color=TUMColor.TUMgreen.value,  # Use TUM green for the remaining trajectory
                marker=".",
                markersize=marksize,
                zorder=35,
                linewidth=lanewidth,
                markeredgewidth=marker_linewidth,
            )
    else:
        if time_step <= tv:
            # Plot the segment from time_step to tv (before TV)
            rnd.ax.plot(
                pos_x[time_step : tv + 1],
                pos_y[time_step : tv + 1],
                color=TUMColor.TUMblue.value,  # Use TUM blue for before TV
                marker="x",
                markersize=marksize,
                zorder=35,
                linewidth=lanewidth,
                markeredgewidth=marker_linewidth,
            )
            # Plot the segment from tv to end_time (after TV)
            rnd.ax.plot(
                pos_x[tv : end_time + 1],
                pos_y[tv : end_time + 1],
                color="red",  # Use red for after TV
                marker="x",
                markersize=marksize,
                zorder=35,
                linewidth=lanewidth,
                markeredgewidth=marker_linewidth,
            )
        else:
            # If time_step > tv, plot only the segment from time_step to end_time
            rnd.ax.plot(
                pos_x[time_step : end_time + 1],
                pos_y[time_step : end_time + 1],
                color="red",  # Use red for the remaining trajectory
                marker="x",
                markersize=marksize,
                zorder=35,
                linewidth=lanewidth,
                markeredgewidth=marker_linewidth,
            )

    if target_veh and world:
        ego_veh_state_ini = ego_to_plot.state_at_time(time_step)
        tar_veh_state = target_veh.state_at_time(time_step)
        tar_veh_lane = world.vehicle_by_id(target_veh.obstacle_id).get_lane(time_step)
        unsafe_poly_ini = compute_unsafe_polygon(
            ego_veh_state_ini, tar_veh_state, target_veh, tar_veh_lane
        )
        rnd.ax.fill(
            *unsafe_poly_ini.exterior.xy,
            zorder=30,
            alpha=0.2,
            facecolor=TUMColor.TUMorange.value,
            edgecolor=None,
        )

    # show plot
    ax.set_xticks([])
    ax.set_yticks([])
    if plot_limits:
        ax.set_xlim([plot_limits[0], plot_limits[1]])
        ax.set_ylim([plot_limits[2], plot_limits[3]])

    if save_path is not None:
        plt.savefig(
            f"{save_path}/{str(scenario.scenario_id)}_{time_step}_once.svg",
            format="svg",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.show(block=True)
    plt.close(fig)


def compute_unsafe_polygon(ego_veh_state, tar_veh_state, target_veh, tar_veh_lane):
    safe_distance = calculate_safe_distance(
        ego_veh_state.velocity, tar_veh_state.velocity, -10.5, -10.0, 0.4
    )
    tar_pos_rear_CART = [
        tar_veh_state.position[0] - target_veh.obstacle_shape.length / 2,
        tar_veh_state.position[1],
    ]
    tar_pos_rear_CVLN = tar_veh_lane.clcs.convert_to_curvilinear_coords(
        tar_pos_rear_CART[0], tar_pos_rear_CART[1]
    )
    safe_pos_CVLN = tar_pos_rear_CVLN - [safe_distance, 0.0]
    safe_pos_CART = tar_veh_lane.clcs.convert_to_cartesian_coords(
        safe_pos_CVLN[0], safe_pos_CVLN[1]
    )

    # left vertices
    tar_pos_rear_left_CART = tar_veh_lane.clcs_left.convert_to_cartesian_coords(
        tar_pos_rear_CVLN[0], 0.0
    )
    safe_pos_left_CART = tar_veh_lane.clcs_left.convert_to_cartesian_coords(
        safe_pos_CVLN[0], 0.0
    )
    ref_left = np.vstack(tar_veh_lane.clcs_left.reference_path())
    vertices_left = ref_left[
        (ref_left[:, 0] > safe_pos_left_CART[0])
        & (ref_left[:, 0] < tar_pos_rear_left_CART[0]),
        :,
    ]
    vertices_left = np.concatenate(
        ([safe_pos_left_CART], vertices_left, [tar_pos_rear_left_CART])
    )

    # right vertices
    tar_pos_rear_right_CART = tar_veh_lane.clcs_right.convert_to_cartesian_coords(
        tar_pos_rear_CVLN[0], 0.0
    )
    safe_pos_right_CART = tar_veh_lane.clcs_right.convert_to_cartesian_coords(
        safe_pos_CVLN[0], 0.0
    )
    ref_right = np.vstack(tar_veh_lane.clcs_right.reference_path())
    vertices_right = ref_right[
        (ref_right[:, 0] > safe_pos_right_CART[0])
        & (ref_right[:, 0] < tar_pos_rear_right_CART[0]),
        :,
    ]
    vertices_right = np.concatenate(
        ([safe_pos_right_CART], vertices_right, [tar_pos_rear_right_CART])
    )

    # the polygon vertices
    vertices_total = np.concatenate(
        (
            [safe_pos_CART],
            vertices_left,
            [tar_pos_rear_CART],
            np.flip(vertices_right, 0),
            [safe_pos_CART],
        )
    ).tolist()
    return Polygon(vertices_total)
