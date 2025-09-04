import math
import re
from fractions import Fraction
import numpy as np
from typing import List, Union, Dict
from collections import defaultdict

from commonroad.scenario.trajectory import Trajectory

from commonroad_crime.utility.simulation import Maneuver
import shapely

from crrepairer.cut_off.tc import TC
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode

# class from STL monitor
from crmonitor.predicates.base import BasePredicateEvaluator
from crmonitor.predicates.position import (
    PredSafeDistPrec,
    PredInSameLane,
    PredInFrontOf,
    PredPreceding,
    PredStopLineInFront,
    PredInIntersectionConflictArea,
    PredOnLaneletWithTypeIntersection,
)
from crmonitor.predicates.velocity import (
    PredLaneSpeedLimit,
    PredFovSpeedLimit,
    PredBrSpeedLimit,
    PredTypeSpeedLimit,
)
from crmonitor.predicates.acceleration import PredAbruptBreaking, PredRelAbruptBreaking
from crmonitor.predicates.general import PredCutIn

from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle
from commonroad.scenario.lanelet import LaneletType
from shapely.geometry import Polygon, LineString

from crrepairer.miqp_planner.configuration import PlanningConfigurationVehicle
from crrepairer.miqp_planner.trajectory import Trajectory as QPTrajectory


class TIConstraint:
    # longitudinal states x
    x_min = -np.inf
    x_max = np.inf
    v_x_min = 0.0
    v_x_max = 60.0
    a_x_min = -10.0
    a_x_max = 12.0
    j_x_min = -15.0
    j_x_max = 15.0
    # longitudinal control u_x
    j_dot_x_min = -5000.0
    j_dot_x_max = 5000.0
    # lateral states y
    d_min = -1000.0
    d_max = 1000.0
    theta_min = -1000.0
    theta_max = 1000.0
    kappa_min = -0.5
    kappa_max = 0.5
    kappa_dot_min = -0.4
    kappa_dot_max = 0.4
    # lateral control
    kappa_dot_dot_min = -100.0
    kappa_dot_dot_max = 100.0
    # slack variable
    slack_min = 0.0
    slack_max = 5000.0
    # react time
    react_time = 0.4
    # length = 4.508
    # width = 1.610
    # wheelbase = 2.578


class LongitudinalConstraint:
    def __init__(self, select_proposition):
        self.rule_constraints: Dict[
            BasePredicateEvaluator.predicate_name, PredicateConstraint
        ] = {}
        self.collision_free_constraints = CollisionFreeConstraint()
        self.tc = None
        self.select_proposition = select_proposition


class LateralConstraint:
    def __init__(self, select_proposition):
        self.d_min = None
        self.d_max = None
        self.long_traj = None
        self.select_proposition = select_proposition


class PredicateConstraint:
    def __init__(
        self,
        decision_variable: bool,
        num_decision_variables: int,
        constraint_state: int,
        constraint_name: str,
        start_time_step: int,
        end_time_step: int,
    ):
        """
        decision_variable: whether decision variables need to be added
        num_decision_variables: number of decision variables to be added
        constraint_state: constraints on which state
        constraint_name: constraint name
        start_time_step: start time step
        end_time_step: end time step
        """
        self.decision_variable = decision_variable
        self.num_decision_variables = num_decision_variables
        self.constraint_state = constraint_state
        self.constraint_name = constraint_name
        self.time_step = list(range(start_time_step, end_time_step + 1))
        self.state_ub = [np.inf] * (end_time_step + 1 - start_time_step)
        self.state_lb = [-np.inf] * (end_time_step + 1 - start_time_step)


class CollisionFreeConstraint:
    def __init__(self):
        self.time_step_lb = list()
        self.time_step_ub = list()
        self.lb = list()
        self.ub = list()


class RuleConstraint:
    def __init__(
        self,
        tc_object: TC,
        rule_monitor: STLRuleMonitor,
        sel_proposition_full: List[PropositionNode],
        proposition_full: List[PropositionNode],
        veh_config: PlanningConfigurationVehicle,
        initial_trajectory: Trajectory,
        start_time_step: int,
    ):
        self.repaired_rules = None
        self._tc_obj = tc_object
        self._rule_monitor = rule_monitor
        self._start_time_step = start_time_step
        self._world_state = self._rule_monitor.world
        self._other_id = self._rule_monitor.other_id
        self._ego_id = (
            self._rule_monitor.vehicle_id
        )  # if no target vehicle, the other_id stands for the ego
        self._ego_vehicle = self._world_state.vehicle_by_id(self._ego_id)
        self._ini_traj = initial_trajectory
        self._target_vehicle: Vehicle = self._world_state.vehicle_by_id(self._other_id)
        self._compliant_maneuver = tc_object.compliant_maneuver
        self._sel_prop_full = sel_proposition_full
        self._prop_full = proposition_full
        self._veh_config = veh_config
        for proposition in proposition_full:
            if "in_intersection_conflict_area__0_1" in proposition.name:
                (
                    self.s_circle_center_front,
                    self.s_circle_center_rear,
                ) = self.create_conflict_area_parameter()
                break

        self._target_lanes = defaultdict(List[Lane])
        self.longitudinal_constraints = LongitudinalConstraint(self._sel_prop_full)
        self.lateral_constraints = LateralConstraint(self._sel_prop_full)

        # safe_dis_mode
        self._safe_dis_mode = [
            False for _ in range(self._tc_obj.N - self._tc_obj.tc_time_step + 1)
        ]
        # time_leave_lane
        if self._compliant_maneuver in [Maneuver.STEERLEFT, Maneuver.STEERRIGHT]:
            # time for leaving the current lane
            self._tc_obj.simulation_lateral.set_inputs(
                self._ego_vehicle.state_list_cr[tc_object.tc_time_step]
            )
            lane_dist = (
                self._ego_vehicle.get_lane(tc_object.tc_time_step).width(
                    self._ego_vehicle.get_lon_state(self._tc_obj.tc_time_step).s
                )
                / 2
                - abs(self._ego_vehicle.get_lat_state(0).d)
                - self._veh_config.width / 2
            )
            leave_time = np.sqrt(
                2 * abs(lane_dist / self._tc_obj.simulation_lateral.a_lat)
            )
            self._time_leave_lane = int(leave_time / self._world_state.dt)

    def construct_longitudinal_constraints(self, veh_config, tc_time_step):
        self.add_rule_constraints()
        self.add_collision_free_constraints()
        self.longitudinal_constraints.tc = tc_time_step

    @property
    def target_lanes(self) -> dict:
        return self._target_lanes

    @property
    def time_leave_lane(self):
        return self._time_leave_lane

    @property
    def safe_distance_modes(self):
        return self._safe_dis_mode

    @property
    def target_vehicle(self) -> Vehicle:
        return self._target_vehicle

    @property
    def sel_prop_full(self) -> list:
        return self._sel_prop_full

    def add_rule_constraints(self):
        # TODO: simplify the index of time step
        num_time_step = 0
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            num_time_step += 1
            total_assignment = self._rule_monitor.prop_robust_all[
                :,
                min(
                    k - self._start_time_step + self._tc_obj.future_time_step,
                    self._tc_obj.N - self._start_time_step,
                ),  # to avoid the index + future time step exceeds the range of `prop_robust_all`
            ]
            for idx, proposition in enumerate(self._rule_monitor.proposition_nodes):
                try:
                    prop_assignment = total_assignment[
                        total_assignment == total_assignment
                    ][idx]
                except:
                    # no assignment can be found
                    continue
                if (
                    proposition in self._prop_full
                    and k
                    >= self._tc_obj.tv_time_step
                    - self._tc_obj.future_time_step  # add constraints only for k>=TV
                ):
                    # proposition to be repaired (greater than the time-to-violation)
                    robs_at_tv = self._rule_monitor.prop_robust_all[
                        :, self._tc_obj.tv_time_step - self._start_time_step
                    ]
                    prop_assignment = robs_at_tv[robs_at_tv == robs_at_tv][idx]
                    if proposition in self._sel_prop_full:
                        prop_assignment = -prop_assignment
                if (
                    k < self._tc_obj.tv_time_step - self._tc_obj.future_time_step
                    or proposition in self._prop_full
                ):
                    # constraints for future temporal operators
                    if (
                        proposition.name[0:5] == "once["
                        and proposition.name[5:6] != proposition.name[7:8]
                    ):
                        pattern = r"once\[(.*?)\]"
                        matches = re.findall(pattern, proposition.name)
                        # get time horizon in future temporal operators
                        further_time = Fraction(matches[0].split(",")[1])
                        future_time_step = int(
                            float(further_time) * self._tc_obj.future_time_step
                        )
                        for time_step in range(
                            k, min(k + future_time_step + 1, self._tc_obj.N + 1)
                        ):
                            self.add_rule_constraint_time_step(
                                proposition, prop_assignment, time_step
                            )
                    else:
                        self.add_rule_constraint_time_step(
                            proposition, prop_assignment, k
                        )

    def add_rule_constraint_time_step(
        self, proposition, prop_assignment, time_step: int
    ):
        # TODO: simplify coding
        for predicate in proposition.children:
            if not hasattr(predicate, "base_name"):
                continue
            if predicate.base_name == PredInSameLane.predicate_name:
                self.ConstrInSameLane(time_step, prop_assignment)
            elif predicate.base_name == PredInFrontOf.predicate_name:
                s_lb, s_ub = self.ConstrInFrontOf(time_step, prop_assignment)
                if (
                    predicate.base_name
                    in self.longitudinal_constraints.rule_constraints.keys()
                ):
                    self._get_overlap(predicate.base_name, s_ub, s_lb, time_step)
                else:
                    self.longitudinal_constraints.rule_constraints[
                        predicate.base_name
                    ] = PredicateConstraint(
                        decision_variable=False,
                        num_decision_variables=0,
                        constraint_state=0,
                        constraint_name="in_front_of",
                        start_time_step=self._tc_obj.tc_time_step,
                        end_time_step=self._tc_obj.N,
                    )
                    self._get_overlap(predicate.base_name, s_ub, s_lb, time_step)
            elif predicate.base_name == PredPreceding.predicate_name:
                # precedes = in_front_of and in_same_lane
                self.ConstrInSameLane(time_step, prop_assignment)
                s_lb, s_ub = self.ConstrInFrontOf(time_step, prop_assignment)
                if (
                    predicate.base_name
                    in self.longitudinal_constraints.rule_constraints.keys()
                ):
                    self._get_overlap(predicate.base_name, s_ub, s_lb, time_step)
                else:
                    self.longitudinal_constraints.rule_constraints[
                        predicate.base_name
                    ] = PredicateConstraint(
                        decision_variable=False,
                        num_decision_variables=0,
                        constraint_state=0,
                        constraint_name="precedes",
                        start_time_step=self._tc_obj.tc_time_step,
                        end_time_step=self._tc_obj.N,
                    )
                    self._get_overlap(predicate.base_name, s_ub, s_lb, time_step)
            elif predicate.base_name == PredSafeDistPrec.predicate_name:
                self.ConstrSafeDist(time_step, prop_assignment)
                # no overlapping is calculated here as it is a nonlinear constraint
            elif predicate.base_name == PredCutIn.predicate_name:
                self.ConstrCutIn(time_step, prop_assignment)
            elif predicate.base_name in (
                PredFovSpeedLimit.predicate_name,
                PredBrSpeedLimit.predicate_name,
                PredTypeSpeedLimit.predicate_name,
                PredLaneSpeedLimit.predicate_name,
            ):
                speed_limit = predicate.evaluator.get_speed_limit(
                    self._world_state, time_step, [self._ego_id]
                )
                if speed_limit is None:
                    speed_limit = np.inf
                v_lb, v_ub = self.ConstrSpeedLimit(speed_limit)
                if (
                    predicate.base_name
                    in self.longitudinal_constraints.rule_constraints.keys()
                ):
                    # existed constraints, then just obtain the overlapped interval
                    self._get_overlap(predicate.base_name, v_ub, v_lb, time_step)
                else:
                    self.longitudinal_constraints.rule_constraints[
                        predicate.base_name
                    ] = PredicateConstraint(
                        decision_variable=False,
                        num_decision_variables=0,
                        constraint_state=1,
                        constraint_name="speed_limit",
                        start_time_step=self._tc_obj.tc_time_step,
                        end_time_step=self._tc_obj.N,
                    )
                    self._get_overlap(predicate.base_name, v_ub, v_lb, time_step)
            elif predicate.base_name in (
                PredAbruptBreaking.predicate_name,
                PredRelAbruptBreaking.predicate_name,
            ):
                a_abruptly = predicate.evaluator.config["a_abrupt"]
                a_lb, a_ub = self.ConstrAccNotAbruptly(a_abruptly)
                if (
                    predicate.base_name
                    in self.longitudinal_constraints.rule_constraints.keys()
                ):
                    self._get_overlap(predicate.base_name, a_ub, a_lb, time_step)
                else:
                    self.longitudinal_constraints.rule_constraints[
                        predicate.base_name
                    ] = PredicateConstraint(
                        decision_variable=False,
                        num_decision_variables=0,
                        constraint_state=2,
                        constraint_name="abrupt_braking",
                        start_time_step=self._tc_obj.tc_time_step,
                        end_time_step=self._tc_obj.N,
                    )
                    self._get_overlap(predicate.base_name, a_ub, a_lb, time_step)

            elif (
                predicate.base_name == PredInIntersectionConflictArea.predicate_name
                and predicate.agent_placeholders == (0, 1)
            ):
                (
                    s_limit_front,
                    s_limit_behind,
                ) = self.ConstrInIntersectionConflictAreaEgo(time_step, prop_assignment)
                if (
                    predicate.base_name
                    in self.longitudinal_constraints.rule_constraints.keys()
                ):
                    self._get_overlap(
                        predicate.base_name, s_limit_front, s_limit_behind, time_step
                    )
                else:
                    self.longitudinal_constraints.rule_constraints[
                        predicate.base_name
                    ] = PredicateConstraint(
                        decision_variable=True,
                        num_decision_variables=1,
                        constraint_state=0,
                        constraint_name="conflict_area",
                        start_time_step=self._tc_obj.tc_time_step,
                        end_time_step=self._tc_obj.N,
                    )
                    self._get_overlap(
                        predicate.base_name, s_limit_front, s_limit_behind, time_step
                    )
            elif predicate.base_name == PredStopLineInFront.predicate_name:
                s_limit_front, s_limit_behind = self.ConstrStopLineInFront(
                    time_step, prop_assignment
                )
                if (
                    predicate.base_name
                    in self.longitudinal_constraints.rule_constraints.keys()
                ):
                    self._get_overlap(
                        predicate.base_name, s_limit_front, s_limit_behind, time_step
                    )
                else:
                    self.longitudinal_constraints.rule_constraints[
                        predicate.base_name
                    ] = PredicateConstraint(
                        decision_variable=False,
                        num_decision_variables=0,
                        constraint_state=0,
                        constraint_name="stop_line",
                        start_time_step=self._tc_obj.tc_time_step,
                        end_time_step=self._tc_obj.N,
                    )
                    self._get_overlap(
                        predicate.base_name, s_limit_front, s_limit_behind, time_step
                    )

    def _get_overlap(self, predicate_name, ub, lb, time_step):
        index = self.longitudinal_constraints.rule_constraints[
            predicate_name
        ].time_step.index(time_step)
        self.longitudinal_constraints.rule_constraints[predicate_name].state_ub[
            index
        ] = min(
            self.longitudinal_constraints.rule_constraints[predicate_name].state_ub[
                index
            ],
            ub,
        )
        self.longitudinal_constraints.rule_constraints[predicate_name].state_lb[
            index
        ] = max(
            self.longitudinal_constraints.rule_constraints[predicate_name].state_lb[
                index
            ],
            lb,
        )

    def add_collision_free_constraints(self):
        if self._rule_monitor.scenario_type == "interstate":
            self.add_collision_free_interstate()
        else:
            self.add_collision_free_intersection()

    def add_collision_free_interstate(self):
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            if k in self._target_lanes:
                try:
                    self._prec_veh, self._foll_veh = self._determine_related_veh(
                        k, self._target_lanes[k]
                    )
                except:
                    self._prec_veh, self._foll_veh = None, None
            else:
                lanelet = (
                    self._world_state.scenario.lanelet_network.find_lanelet_by_position(
                        [self._ego_vehicle.states_cr[k].position]
                    )[0]
                )
                lanes = self._world_state.road_network.find_lanes_by_lanelets(
                    set(lanelet)
                )
                if lanes:
                    self._prec_veh, self._foll_veh = self._determine_related_veh(
                        k, list(lanes)
                    )
            index = k - self._tc_obj.tc_time_step
            if self._prec_veh is not None:
                if k <= self._prec_veh.end_time:
                    self.longitudinal_constraints.collision_free_constraints.time_step_ub.append(
                        index
                    )
                    self.longitudinal_constraints.collision_free_constraints.ub.append(
                        self._prec_veh.rear_s(k, self._ego_vehicle.ref_path_lane)
                        - self._veh_config.wheelbase / 2
                        - self._veh_config.length / 2
                    )
            # if self._foll_veh is not None:
            #     self.longitudinal_constraints.collision_free_constraints.time_step_lb.append(
            #         index
            #     )
            #     self.longitudinal_constraints.collision_free_constraints.lb.append(
            #         self._prec_veh.front_s(k, self._ego_vehicle.ref_path_lane)
            #         + self._veh_config.wheelbase / 2
            #         + self._veh_config.length / 2
            #     )

    def add_collision_free_intersection(self):
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            # todo: add target lanes
            if k in self._target_lanes and None not in self._target_lanes[k]:
                lanes = [self._ego_vehicle.ref_path_lane] + self._target_lanes[k]
            else:
                lanes = [self._ego_vehicle.ref_path_lane]
            self._prec_veh, self._foll_veh = self._determine_related_veh(k, lanes)
            index = k - self._tc_obj.tc_time_step
            if self._prec_veh is not None:
                if k <= self._prec_veh.end_time:
                    conflict_points = self.calculation_circle_approximation(k)
                    if conflict_points is not None:
                        s_min = self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                            *conflict_points[0]
                        )[
                            0
                        ]
                        self.longitudinal_constraints.collision_free_constraints.time_step_ub.append(
                            index
                        )
                        self.longitudinal_constraints.collision_free_constraints.ub.append(
                            s_min
                            - self._ego_vehicle.shape.length / 3
                            - self._veh_config.wheelbase / 2
                        )

    def calculation_circle_approximation(self, time_step):
        circle_target = self._prec_veh.circle_appr_occupancy_at_time_step(time_step)
        offset_circle_target = shapely.offset_curve(
            circle_target, self._ego_vehicle.circle_radius
        )
        reference_lane_center = LineString(
            self._ego_vehicle.ref_path_lane.smoothed_vertices
        )
        intersection = reference_lane_center.intersection(offset_circle_target)
        conflict_line_points = list()
        if intersection.geom_type == "Point":
            conflict_line_points.append([intersection.x, intersection.y])
        elif (
            intersection.geom_type == "LineString"
            or intersection.geom_type == "LinearRing"
        ):
            for point in intersection.coords:
                conflict_line_points.append(np.array(point))
        elif (
            intersection.geom_type == "MultiPoint"
            or intersection.geom_type == "MultiLineString"
        ):
            for geom in intersection.geoms:
                for point in geom.coords:
                    conflict_line_points.append(point)
        if len(conflict_line_points) == 0:
            conflict_points = None
        else:
            conflict_points = [conflict_line_points[0], conflict_line_points[-1]]
        return conflict_points

    def _determine_related_veh(self, time_step, lanes: List[Lane]):
        preceding_vehicle = None
        following_vehicle = None
        dist_pre = np.inf
        dist_post = -np.inf
        vehicle_ids = set()
        for lane in lanes:
            vehicle_ids.update(lane.lanelet.dynamic_obstacle_by_time_step(time_step))
        vehicle_ids.discard(self._ego_id)
        ego_front_s = self._ego_vehicle.front_s(
            time_step, self._ego_vehicle.ref_path_lane
        )
        for id in vehicle_ids:
            if id in self._world_state.vehicle_ids():
                other_vehicle = self._world_state.vehicle_by_id(id)
            else:
                continue
            if other_vehicle is None:
                continue
            try:
                other_front_s = other_vehicle.front_s(
                    time_step, self._ego_vehicle.ref_path_lane
                )
            except:
                continue
            dist = other_front_s - ego_front_s
            if 0 < dist < dist_pre:
                preceding_vehicle = other_vehicle
                dist_pre = dist
            elif 0 > dist > dist_post:
                following_vehicle = other_vehicle
                dist_post = dist
            else:
                continue
        return preceding_vehicle, following_vehicle

    def ConstrInSameLane(self, time_step: int, prop_assignment: float):
        if self._target_vehicle.id == self._ego_vehicle.id:
            target_lane = [None]
        elif time_step in self._target_vehicle.lanelet_assignment.keys():
            tar_veh_lanelet = self._target_vehicle.lanelet_assignment[time_step]
            try:
                # todo fix target lane in intersection
                if self._rule_monitor.scenario_type == "interstate":
                    tar_veh_lane = self._world_state.road_network.find_lane_by_lanelet(
                        list(tar_veh_lanelet)[0]
                    )
                else:
                    tar_veh_lane = self._target_vehicle.ref_path_lane

                # if prop_assignment > 0:
                #    target_lane = [tar_veh_lane]
                if self._compliant_maneuver == Maneuver.STEERLEFT:
                    target_lane = [tar_veh_lane.adj_right]
                elif self._compliant_maneuver == Maneuver.STEERRIGHT:
                    target_lane = [tar_veh_lane.adj_left]
                else:
                    target_lane = [tar_veh_lane]
                if self._compliant_maneuver in [
                    Maneuver.STEERLEFT,
                    Maneuver.STEERRIGHT,
                ]:
                    if time_step <= self._time_leave_lane:
                        target_lane = [tar_veh_lane]
                    elif self._time_leave_lane < time_step <= self._tc_obj.tv_time_step:
                        target_lane += [tar_veh_lane]
                    target_lane = sorted(target_lane, key=lambda lane: lane.lane_id)
            except:
                target_lane = [None]

        else:
            target_lane = [None]
        self._target_lanes[time_step] = list(set(target_lane))

    def ConstrInFrontOf(self, time_step: int, prop_assignment: float):
        if self._target_vehicle.id == self._ego_vehicle.id:
            return -np.inf, np.inf
        # preventing KeyError
        if time_step > self._target_vehicle.end_time:
            return -np.inf, np.inf
        if prop_assignment > 0:
            rear_s = self._target_vehicle.rear_s(time_step)
            if rear_s:
                return -np.inf, rear_s
            else:
                return -np.inf, np.inf
        else:
            front_s = self._target_vehicle.front_s(time_step)
            if front_s:
                return front_s, np.inf
            else:
                return -np.inf, np.inf

    def ConstrSafeDist(self, time_step: int, prop_assignment: float):
        if prop_assignment > 0:
            self._safe_dis_mode[time_step - self._tc_obj.tc_time_step] = True
        else:
            pass

    def ConstrCutIn(
        self,
        time_step: int,
        prop_assignment: float,
    ):
        # print("<QPRepairer/_rule_constraints>: we cannot add constraints for cut in")
        return None

    def ConstrSpeedLimit(self, speed_limit):
        # todo: repair failed because of speed constraint
        return 0, speed_limit - 1e-1

    def ConstrAccNotAbruptly(self, a_abrupt):
        return a_abrupt, np.inf

    def ConstrStopLineInFront(self, time_step: int, prop_assignment: float):
        wold = self._rule_monitor.world
        upper_bound = np.inf
        for lanelet_id in self._ego_vehicle.lanelets_dir:
            lanelet = wold.road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            if lanelet.stop_line is not None:
                stop_line_s = min(
                    self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                        *lanelet.stop_line.start
                    )[0],
                    self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                        *lanelet.stop_line.end
                    )[0],
                )
                upper_bound = min(
                    upper_bound,
                    stop_line_s
                    - self._ego_vehicle.circle_radius
                    - self._veh_config.length / 3
                    - self._veh_config.wheelbase / 2,
                )
        return upper_bound, -math.inf

    def ConstrOnLaneletWithTypeIntersection(
        self, time_step: int, prop_assignment: float
    ):
        if prop_assignment > 0:
            return math.inf, -math.inf
        else:
            incoming = self._ego_vehicle.incoming_intersection
            turning_lanelets = incoming.successors_left.union(
                incoming.successors_right,
                incoming.successors_straight,
            )
            lanelet_intersection_id = list(
                self._ego_vehicle.ref_path_lane.contained_lanelets.intersection(
                    turning_lanelets
                )
            )
            lanelet_intersection = self._rule_monitor.world.road_network.lanelet_network.find_lanelet_by_id(
                lanelet_intersection_id[0]
            )
            start_s = min(
                self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *lanelet_intersection.right_vertices[0]
                )[0],
                self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *lanelet_intersection.left_vertices[0]
                )[0],
            )
            end_s = max(
                self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *lanelet_intersection.right_vertices[-1]
                )[0],
                self._ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *lanelet_intersection.left_vertices[-1]
                )[0],
            )
            upper_bound = (
                start_s
                - self._ego_vehicle.circle_radius
                - self._veh_config.length / 3
                - self._veh_config.wheelbase / 2
            )
            lower_bound = end_s + self._ego_vehicle.circle_radius
            return upper_bound, lower_bound

    def ConstrInIntersectionConflictAreaEgo(
        self, time_step: int, prop_assignment: float
    ):
        if prop_assignment <= 0:
            front_constr = (
                self.s_circle_center_front
                - self._ego_vehicle.shape.length / 3
                - self._veh_config.wheelbase / 2
            )
            rear_constr = self.s_circle_center_rear
            return front_constr, rear_constr
        else:
            return math.inf, -math.inf

    def create_conflict_area_parameter(self):
        ego_vehicle = self._ego_vehicle
        target_vehicle = self._target_vehicle
        road_network = self._rule_monitor.world.road_network

        # offset conflict lanelets
        conflict_lanelets_shape = list()
        for lanelet_id in target_vehicle.ref_path_lane.contained_lanelets:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            if LaneletType.INTERSECTION in lanelet.lanelet_type:
                conflict_lanelets_shape.append(lanelet.polygon.shapely_object)
        conflict_area_shape = shapely.unary_union(conflict_lanelets_shape)
        conflict_linestring = shapely.offset_curve(
            conflict_area_shape, ego_vehicle.circle_radius
        )

        # find right conflict point
        line_right = LineString(ego_vehicle.lanelets_dir_right_vertices)
        line_right_offset = shapely.offset_curve(line_right, ego_vehicle.circle_radius)
        conflict_circle_center_right = self.find_conflict_points(
            line_right_offset, conflict_linestring
        )
        if conflict_circle_center_right is not None:
            s_circle_center_right = [
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *conflict_circle_center_right[0]
                )[0],
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *conflict_circle_center_right[1]
                )[0],
            ]
            s_circle_center_right = np.sort(s_circle_center_right)
        else:
            s_circle_center_right = np.array([np.inf, -np.inf])

        # find left conflict point
        line_left = LineString(ego_vehicle.lanelets_dir_left_vertices)
        line_left_offset = shapely.offset_curve(line_left, -ego_vehicle.circle_radius)
        conflict_circle_center_left = self.find_conflict_points(
            line_left_offset, conflict_linestring
        )
        if conflict_circle_center_left is not None:
            s_circle_center_left = [
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *conflict_circle_center_left[0]
                )[0],
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *conflict_circle_center_left[1]
                )[0],
            ]
            s_circle_center_left = np.sort(s_circle_center_left)
        else:
            s_circle_center_left = np.array([np.inf, -np.inf])

        # find center conflict point
        line_center = LineString(ego_vehicle.lanelets_dir_center_vertices)
        conflict_circle_center_center = self.find_conflict_points(
            line_center, conflict_linestring
        )
        if conflict_circle_center_center is not None:
            s_circle_center_center = [
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *conflict_circle_center_center[0]
                )[0],
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *conflict_circle_center_center[1]
                )[0],
            ]
            s_circle_center_center = np.sort(s_circle_center_center)
        else:
            s_circle_center_center = np.array([np.inf, -np.inf])

        # find conflict point for constraints
        s_circle_center_front = np.min(
            [
                s_circle_center_right[0],
                s_circle_center_left[0],
                s_circle_center_center[0],
            ]
        )
        s_circle_center_rear = np.max(
            [
                s_circle_center_right[1],
                s_circle_center_left[1],
                s_circle_center_center[1],
            ]
        )

        return s_circle_center_front, s_circle_center_rear

    @staticmethod
    def find_conflict_points(
        curved_line: LineString, conflict_polygon: Union[Polygon, LineString]
    ):
        conflict_line_points = list()
        # Get intersection of line and polygon
        intersection = curved_line.intersection(conflict_polygon)
        if intersection.geom_type == "Point":
            conflict_line_points.append([intersection.x, intersection.y])
        elif (
            intersection.geom_type == "LineString"
            or intersection.geom_type == "LinearRing"
        ):
            for point in intersection.coords:
                conflict_line_points.append(np.array(point))
        elif (
            intersection.geom_type == "MultiPoint"
            or intersection.geom_type == "MultiLineString"
        ):
            for geom in intersection.geoms:
                for point in geom.coords:
                    conflict_line_points.append(point)
        if len(conflict_line_points) == 0:
            conflict_points = None
        else:
            conflict_points = [conflict_line_points[0], conflict_line_points[-1]]
        return conflict_points

    def create_d_constraints(self, long_traj: QPTrajectory, veh_config):
        self.lateral_constraints.long_traj = long_traj
        # TODO: fix construction (now copy from rule_constraints)
        self._lat_dis_constraints = list()
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            d_min = -np.inf
            d_max = np.inf
            if k in self._target_lanes and None not in self._target_lanes[k]:
                target_lanes = self._target_lanes[k]
                index = k - self._tc_obj.tc_time_step
                lane_boundary_left = target_lanes[
                    -1
                ].clcs_left.convert_to_cartesian_coords(
                    long_traj.states[index].position[0], 0.0
                )
                lane_boundary_right = target_lanes[
                    0
                ].clcs_right.convert_to_cartesian_coords(
                    long_traj.states[index].position[0], 0.0
                )
                d_max = min(
                    self._veh_config.curvilinear_coordinate_system.convert_to_curvilinear_coords(
                        lane_boundary_left[0], lane_boundary_left[1]
                    )[
                        1
                    ],
                    d_max,
                )
                d_min = max(
                    self._veh_config.curvilinear_coordinate_system.convert_to_curvilinear_coords(
                        lane_boundary_right[0], lane_boundary_right[1]
                    )[
                        1
                    ],
                    d_min,
                )
            self._lat_dis_constraints.append([d_min, d_max])
        lateral_constraints = np.array(self._lat_dis_constraints)
        d_min = np.array(
            (
                lateral_constraints[1:, 0],
                lateral_constraints[1:, 0],
                lateral_constraints[1:, 0],
            )
        ).transpose()
        d_max = np.array(
            (
                lateral_constraints[1:, 1],
                lateral_constraints[1:, 1],
                lateral_constraints[1:, 1],
            )
        ).transpose()
        self.lateral_constraints.d_min = d_min
        self.lateral_constraints.d_max = d_max
