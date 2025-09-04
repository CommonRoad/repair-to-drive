from typing import List, Optional


from crrepairer.miqp_planner.trajectory import Trajectory, TrajPoint
from crrepairer.miqp_planner.miqp_lat_planner import MIQPLatState, MIQPLatReference
from crrepairer.miqp_planner.miqp_constraints_manual import (
    LongitudinalConstraint,
    LateralConstraint,
    TIConstraint,
)
from crrepairer.miqp_planner.miqp_long_planner import MIQPLongReference

from crmonitor.common.vehicle import Vehicle
from crrepairer.utils.configuration import RepairerConfiguration


class MIQPPlanner:
    def __init__(self, config: RepairerConfiguration):
        """Base class for MIQP Planner"""
        self.scenario = config.scenario
        self.planning_problem = config.planning_problem
        self.vehicle_configuration = config.vehicle
        if not hasattr(self.scenario, "dt"):
            self.dt = 0.1  # default time step
        else:
            self.dt = self.scenario.dt

        self.initial_state: Optional[TrajPoint] = None
        # self.t_h = config.miqp_planner.horizon
        #
        # if isinstance(self.planning_problem.initial_state, State):
        #     # this state is in curvilinear coordinate system
        #     self.initial_state = compute_initial_state(
        #         self.planning_problem.initial_state, config.vehicle.qp_veh_config
        #     )
        # elif not isinstance(self.planning_problem.initial_state, TrajPoint):
        #     raise ValueError(
        #         "<MIQPPlanner/__init__>: Initial state must be of type {} or "
        #         "of type {}. Got type {}.".format(
        #             type(State),
        #             type(TrajPoint),
        #             type(self.planning_problem.initial_state),
        #         )
        #     )
        # if (
        #     self.vehicle_configuration.qp_veh_config.reference_point
        #     != ReferencePoint.REAR
        # ):
        #     raise ValueError("<MIQPPlanner>: Reference point must be rear axis!")
        #
        # if self.planning_problem.goal.state_list:
        #     if (
        #         self.initial_state.v
        #         > self.planning_problem.goal.state_list[0].velocity.end
        #     ):
        #         self.vehicle_configuration.desired_speed = (
        #             self.planning_problem.goal.state_list[0].velocity.end
        #         )
        #     else:
        #         self.vehicle_configuration.desired_speed = self.initial_state.v
        # else:
        #     self.vehicle_configuration.desired_speed = self.initial_state.v
        #
        # # initial orientation for the lateral planner
        # self.initial_state_lat_orientation = self.initial_state.orientation
        self.config = config

        # set up the time invariant constraints
        self.time_invariant_constraints = TIConstraint()

        # initialize the planner for saving time later on:
        self.long_planner = None
        self.lat_planner = None

    def _set_time_invariant_constraints(self):
        """Sets the time invariant constraints from the configuration file."""
        ti_constraint = TIConstraint()
        ti_constraint.v_x_max = self.config.vehicle.qp_veh_config.max_speed_x
        ti_constraint.v_x_min = self.config.vehicle.qp_veh_config.min_speed_x
        ti_constraint.a_x_max = self.config.vehicle.qp_veh_config.a_max_x
        ti_constraint.a_x_min = self.config.vehicle.qp_veh_config.a_min_x
        ti_constraint.j_x_max = self.config.vehicle.qp_veh_config.j_max_x
        ti_constraint.j_x_min = self.config.vehicle.qp_veh_config.j_min_x

        ti_constraint.kappa_max = self.config.vehicle.kappa_max
        ti_constraint.kappa_min = self.config.vehicle.kappa_min
        ti_constraint.kappa_dot_max = self.config.vehicle.kappa_dot_max
        ti_constraint.kappa_dot_min = self.config.vehicle.kappa_dot_min
        ti_constraint.kappa_dot_dot_max = self.config.vehicle.kappa_dot_dot_max
        ti_constraint.kappa_dot_dot_min = self.config.vehicle.kappa_dot_dot_min

        ti_constraint.react_time = self.config.vehicle.qp_veh_config.react_time
        ti_constraint.length = self.config.vehicle.qp_veh_config.length
        ti_constraint.width = self.config.vehicle.qp_veh_config.width
        ti_constraint.wheelbase = self.config.vehicle.qp_veh_config.wheelbase
        return ti_constraint

    def longitudinal_trajectory_planning(
        self,
        reference_lon: MIQPLongReference,
        long_constraints: LongitudinalConstraint,
        safe_distance_modes: List[bool],
        pred_veh: Vehicle,
    ):
        """Plans the longitudinal trajectory"""
        self.long_planner.reset(self.planning_problem.initial_state)
        self.time_invariant_constraints = self._set_time_invariant_constraints()
        traj_long = self.long_planner.plan(
            reference_lon,
            self.time_invariant_constraints,
            long_constraints,
            safe_distance_modes,
            pred_veh,
        )
        return traj_long

    def lateral_trajectory_planning(
        self,
        longitudinal_trajectory: Trajectory,
        lat_con: LateralConstraint,
    ):
        """Plans the lateral trajectory"""
        x_init_lat = MIQPLatState(
            d=self.initial_state.position[1],
            theta=self.initial_state.orientation,
            kappa=self.initial_state.kappa,
            kappa_dot=self.initial_state.kappa_dot,
            t=0.0,
            s=longitudinal_trajectory.states[0].position[0],
            v=longitudinal_trajectory.states[0].v,
            a=longitudinal_trajectory.states[0].a,
            j=longitudinal_trajectory.states[0].j,
            u_lon=longitudinal_trajectory.u_lon,
        )
        x_ref_lat = MIQPLatReference.construct_from_lon_traj_and_reference(
            lon_traj=longitudinal_trajectory,
            reference=self.vehicle_configuration.qp_veh_config.reference_path,
            vehicle_configuration=self.vehicle_configuration,
        )
        self.lat_planner.reset(x_init_lat=x_init_lat, x_ref_lat=x_ref_lat)
        trajectory = self.lat_planner.plan(
            lateral_constraints=lat_con,
            ti_constraints=self.time_invariant_constraints,
        )
        return trajectory
