from commonroad.prediction.prediction import Trajectory
from commonroad.scenario.state import CustomState, ExtendedPMState, InitialState
from commonroad.common.util import AngleInterval, Interval
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.geometry.shape import Rectangle

from crrepairer.utils.configuration import RepairerConfiguration


def update_goal_state(initial_trajectory: Trajectory):
    """
    Update goal state for the reference generation.
    :return: the updated goal state
    """
    ini_final_state = initial_trajectory.state_list[-1]
    goal_orientation = AngleInterval(
        ini_final_state.orientation - 0.2, ini_final_state.orientation + 0.2
    )
    goal_velocity = Interval(ini_final_state.velocity, ini_final_state.velocity + 5.0)
    goal_time_step = Interval(0, len(initial_trajectory.state_list) + 5)
    goal_state = CustomState(
        position=Rectangle(1, 1, ini_final_state.position),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step,
    )
    goal_region = GoalRegion([goal_state])
    return goal_region


def update_goal_state_extension(
    initial_trajectory: Trajectory, lanelet_network: LaneletNetwork
):
    """
    Update goal state for the reference generation.
    :return: the updated goal state
    """
    ini_final_state = initial_trajectory.state_list[-1]
    ini_final_lanelet = lanelet_network.find_lanelet_by_position(
        [ini_final_state.position]
    )[0]
    for lanelet_id in lanelet_network.find_lanelet_by_id(
        ini_final_lanelet[0]
    ).successor:
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        # if lanelet
    successor_lanelet = lanelet_network.find_lanelet_by_id(
        lanelet_network.find_lanelet_by_id(ini_final_lanelet[0]).successor[0]
    )
    ref_point = successor_lanelet.center_vertices[-1]
    goal_orientation = AngleInterval(
        ini_final_state.orientation - 0.2, ini_final_state.orientation + 0.2
    )
    goal_velocity = Interval(ini_final_state.velocity, ini_final_state.velocity + 5.0)
    goal_time_step = Interval(0, len(initial_trajectory.state_list) + 5)
    goal_state = CustomState(
        position=Rectangle(1, 1, ref_point),
        velocity=goal_velocity,
        orientation=goal_orientation,
        time_step=goal_time_step,
    )
    goal_region = GoalRegion([goal_state])
    return goal_region


def retrieve_ego_vehicle(config: RepairerConfiguration):
    """Retrieves the ego vehicle based on the given time frame."""
    ego_initial = config.scenario.obstacle_by_id(config.repair.ego_id)
    new_state_list = []
    for time_step in range(config.repair.t_0, config.repair.t_f):
        if ego_initial.state_at_time(time_step):
            if isinstance(ego_initial.state_at_time(time_step), ExtendedPMState):
                new_state = CustomState(
                    time_step=ego_initial.state_at_time(time_step).time_step,
                    position=ego_initial.state_at_time(time_step).position,
                    velocity=ego_initial.state_at_time(time_step).velocity,
                    orientation=ego_initial.state_at_time(time_step).orientation,
                    acceleration=ego_initial.state_at_time(time_step).acceleration,
                )
            else:
                new_state = ego_initial.state_at_time(time_step)
            if not isinstance(new_state, InitialState):
                # skip the initial state with different type
                new_state_list.append(new_state)
        else:
            print(f"ego vehicle does not have state at time step {time_step}")
    ego_initial.prediction.trajectory = Trajectory(
        new_state_list[0].time_step, new_state_list
    )
    return ego_initial
