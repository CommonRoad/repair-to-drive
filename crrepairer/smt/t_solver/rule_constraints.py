import numpy as np
from typing import List, Optional, Union
from collections import defaultdict

import shapely

from commonroad.scenario.trajectory import Trajectory
from commonroad.geometry.transform import rotate_translate
from commonroad_crime.utility.simulation import Maneuver

from crrepairer.cut_off.tc import TC
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode

# class from STL monitor
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
from crmonitor.predicates.general import PredCutIn
from crmonitor.predicates.acceleration import PredAbruptBreaking, PredRelAbruptBreaking

from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.road_network import RoadNetwork
from commonroad.scenario.lanelet import LaneletType
from shapely.geometry import Polygon, LineString, Point

from crrepairer.miqp_planner.configuration import PlanningConfigurationVehicle
from crrepairer.miqp_planner.constraints import LonConstraints, LatConstraints
from crrepairer.miqp_planner.initialization import convert_pos_curvilinear
from crrepairer.miqp_planner.trajectory import Trajectory as QPTrajectory


class RuleConstraintsManual:
    """
    Class for traffic rule constraints
    """

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
        # initialize the needed components
        self._tc_obj = tc_object
        self._rule_monitor = rule_monitor
        self._world_state = self._rule_monitor.world
        self._other_id = self._rule_monitor.other_id
        self._ego_id = (
            self._rule_monitor.vehicle_id
        )  # if no target vehicle, the other_id stands for the ego
        self._ego_vehicle = self._world_state.vehicle_by_id(self._ego_id)
        self._ini_traj = initial_trajectory
        self._start_time_step = start_time_step
        self._target_vehicle: Vehicle = self._world_state.vehicle_by_id(self._other_id)
        self._veh_config = veh_config
        self._compliant_maneuver = tc_object.compliant_maneuver
        self._sel_prop_full = sel_proposition_full
        self._prop_full = proposition_full

        # initialize the elements for rule constraints
        self._target_lanes = defaultdict(List[Lane])
        self._lon_dis_constraints = list()
        self._lon_vel_constraints = list()
        self._lon_acc_constraints = list()
        self._lat_dis_constraints = list()

        self._prec_veh = None
        self._foll_veh = None
        # whether safe distance needs to be obeyed
        self._safe_dis_mode = []
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

        # initial conflict area parameter
        self.s_circle_center_front = None
        self.s_circle_center_rear = None
        self.conflict_line_front = None
        self.conflict_line_rear = None

    def reset(
        self,
        start_time_step: int,
        tc_object: TC,
        rule_monitor: STLRuleMonitor,
        sel_proposition_full: List[PropositionNode],
        proposition_full: List[PropositionNode],
    ):
        if start_time_step is not None:
            self._start_time_step = start_time_step

        if rule_monitor is not None:
            self._other_id = rule_monitor.other_id

        if tc_object is not None:
            self._tc_obj = tc_object
            self._compliant_maneuver = tc_object.compliant_maneuver
            self._safe_dis_mode = [
                False for _ in range(tc_object.N - tc_object.tc_time_step + 1)
            ]
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

        if sel_proposition_full is not None:
            self._sel_prop_full = sel_proposition_full

        if proposition_full is not None:
            self._prop_full = proposition_full
            for proposition in proposition_full:
                if "in_intersection_conflict_area__0_1" in proposition.name:
                    (
                        self.s_circle_center_front,
                        self.s_circle_center_rear,
                    ) = self.create_conflict_area_parameter()
                    break

    @property
    def safe_distance_modes(self):
        return self._safe_dis_mode

    @property
    def target_lanes(self) -> dict:
        return self._target_lanes

    @property
    def time_leave_lane(self):
        return self._time_leave_lane

    def add(self):
        """
        add rule constraints. Since QP planner is used for longitudinal and lateral motions separately,
        we can only first obtain the numerical values for then longitudinal motion and the lane constraints
        for the lateral motion.
            longitudinal motion: s, v, a
            lateral motion: lane
        """
        # TODO: consider future temporal operators
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            total_assignment = self._rule_monitor.prop_robust_all[
                :, k - self._start_time_step
            ]
            # longitudinal position and velocity limit
            s_limit = [-np.inf, np.inf]
            v_limit = [0, np.inf]
            a_limit = [-np.inf, np.inf]
            for idx, proposition in enumerate(self._rule_monitor.proposition_nodes):
                try:
                    prop_assignment = total_assignment[
                        total_assignment == total_assignment
                    ][idx]
                except:
                    # no assignment can be found
                    continue
                for predicate in proposition.children:
                    if (
                        proposition in self._prop_full
                        and k >= self._tc_obj.tv_time_step
                    ):
                        # proposition to be repaired (greater than the time-to-violation)
                        robs_at_tv = self._rule_monitor.prop_robust_all[
                            :, self._tc_obj.tv_time_step
                        ]
                        prop_assignment = robs_at_tv[robs_at_tv == robs_at_tv][idx]
                        if proposition in self._sel_prop_full:
                            prop_assignment = -prop_assignment
                    if k < self._tc_obj.tv_time_step or proposition in self._prop_full:
                        if not hasattr(predicate, "base_name"):
                            continue
                        if predicate.base_name == PredInSameLane.predicate_name:
                            self.ConstrInSameLane(k, prop_assignment)
                        elif predicate.base_name == PredInFrontOf.predicate_name:
                            s_constr = self.ConstrInFrontOf(k, prop_assignment)
                            s_limit = self._get_overlap(s_limit, s_constr)
                        elif predicate.base_name == PredPreceding.predicate_name:
                            # precedes = in_front_of  and in_same_lane
                            self.ConstrInSameLane(k, prop_assignment)
                            s_constr = self.ConstrInFrontOf(k, prop_assignment)
                            s_limit = self._get_overlap(s_limit, s_constr)
                        elif predicate.base_name == PredSafeDistPrec.predicate_name:
                            self.ConstrSafeDist(k, prop_assignment)
                        elif predicate.base_name == PredCutIn.predicate_name:
                            self.ConstrCutIn(k, prop_assignment)
                        elif predicate.base_name in (
                            PredFovSpeedLimit.predicate_name,
                            PredBrSpeedLimit.predicate_name,
                            PredTypeSpeedLimit.predicate_name,
                            PredLaneSpeedLimit.predicate_name,
                        ):
                            speed_limit = predicate.evaluator.get_speed_limit(
                                self._world_state, k, [self._ego_id]
                            )
                            if speed_limit is None:
                                speed_limit = np.inf
                            v_constr = self.ConstrSpeedLimit(speed_limit)
                            v_limit = self._get_overlap(v_limit, v_constr)
                        elif predicate.base_name in (
                            PredAbruptBreaking.predicate_name,
                            PredRelAbruptBreaking.predicate_name,
                        ):
                            a_abruptly = predicate.evaluator.config["a_abrupt"]
                            a_constr = self.ConstrAccNotAbruptly(a_abruptly)
                            a_limit = self._get_overlap(a_constr, a_limit)
                        # --------------------------------------------------------------------------------------------#
                        elif predicate.base_name in (
                            PredStopLineInFront.predicate_name,
                        ):
                            s_constr = self.ConstrStopLine(k, prop_assignment)
                            s_limit = self._get_overlap(s_limit, s_constr)
                        elif predicate.base_name in (
                            PredInIntersectionConflictArea.predicate_name,
                        ):
                            s_constr = self.ConstrInIntersectionConflictAreaEgo(
                                k, prop_assignment
                            )
                            s_limit = self._get_overlap(s_limit, s_constr)
                        else:
                            print(
                                "<QPRepairer/_rule_constraints>: the provided predicate {} "
                                "is not supported".format(predicate.name)
                            )
            self._lon_dis_constraints.append(s_limit)
            self._lon_vel_constraints.append(v_limit)
            self._lon_acc_constraints.append(a_limit)

    def longitudinal_constraints(self, vehicle_configuration):
        """
        Set the longitudinal constraints
        """
        # add general constraints
        self.add()
        # add collision constraints (implicitly)
        if self._rule_monitor.scenario_type == "interstate":
            self.ConstrCollisionFree()
        else:
            self.ConstrCollisionFreeIntersection()
        longitudinal_distance_constraints = np.array(self._lon_dis_constraints)
        longitudinal_velocity_constraints = np.array(self._lon_vel_constraints)
        longitudinal_acceleration_constraints = np.array(self._lon_acc_constraints)
        return LonConstraints.construct_constraints(
            longitudinal_distance_constraints[1:, 0],
            longitudinal_distance_constraints[1:, 1],
            longitudinal_distance_constraints[1:, 0],
            longitudinal_distance_constraints[1:, 1],
            v_min=longitudinal_velocity_constraints[1:, 0],
            v_max=longitudinal_velocity_constraints[1:, 1],
            a_min=longitudinal_acceleration_constraints[1:, 0],
            a_max=longitudinal_acceleration_constraints[1:, 1],
            prec_veh=self._target_vehicle,
            tc_time_step=self._tc_obj.tc_time_step,
            select_proposition=self._sel_prop_full,
        )

    def lateral_constraints(self, long_traj: QPTrajectory, configuration_qp=None):
        """
        Set the lateral constraints (based on the planned longitudinal trajectory and the previously
        added lane constraints - target lanes)
        """
        self._lat_dis_constraints = []
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            d_min = -np.inf
            d_max = np.inf
            if k in self._target_lanes:
                target_lanes = self._target_lanes[k]
                if None not in target_lanes:
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
                        self._veh_config.CLCS.convert_to_curvilinear_coords(
                            lane_boundary_left[0], lane_boundary_left[1]
                        )[1],
                        d_max,
                    )
                    d_min = max(
                        self._veh_config.CLCS.convert_to_curvilinear_coords(
                            lane_boundary_right[0], lane_boundary_right[1]
                        )[1],
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
        return LatConstraints.construct_constraints(d_min, d_max, d_min, d_max)

    def _determine_related_veh(self, time_step: int, lanes: List[Lane]):
        """
        Determines the related vehicles for collision-free constraints
        """
        preceding_vehicle = None
        following_vehicle = None
        dist_pre = np.inf
        dist_post = -np.inf
        vehicle_ids = set()
        if not list(lanes)[0]:
            return None, None
        # find all the vehicles in the target lanes (then remove the ego)
        for lane in lanes:
            vehicle_ids.update(lane.lanelet.dynamic_obstacle_by_time_step(time_step))
        vehicle_ids.discard(self._ego_id)
        for id in vehicle_ids:
            other_vehicle = self._world_state.vehicle_by_id(id)
            if other_vehicle is None:
                continue
            if time_step == 0:
                ego_state = self._ego_vehicle.states_cr[0]
            else:
                ego_state = self._ini_traj.state_at_time_step(time_step)
            ego_lon_s = convert_pos_curvilinear(ego_state, self._veh_config)[0]
            # if time_step in other_vehicle.states_lon:
            #    dist = other_vehicle.states_lon[time_step].s - ego_lon_s
            # else:
            #    continue
            try:
                dist = other_vehicle.get_lon_state(time_step).s - ego_lon_s
            except:
                continue
            if 0 < dist < dist_pre:
                preceding_vehicle = other_vehicle
                dist_pre = dist
            elif 0 > dist > dist_post:
                following_vehicle = other_vehicle
                dist_post = dist
            else:
                continue
        return preceding_vehicle, following_vehicle

    def _determine_related_veh_intersection(self, time_step: int):
        """
        Determines the related vehicles for collision-free constraints in intersection scenarios
        """
        preceding_vehicle = None
        following_vehicle = None
        dist_pre = np.inf
        dist_post = -np.inf
        vehicle_ids = set()
        lane = self._ego_vehicle.ref_path_lane
        vehicle_ids.update(lane.lanelet.dynamic_obstacle_by_time_step(time_step))
        vehicle_ids.discard(self._ego_id)
        for id in vehicle_ids:
            other_vehicle = self._world_state.vehicle_by_id(id)
            if other_vehicle is None:
                continue
            if time_step == 0:
                ego_state = self._ego_vehicle.states_cr[0]
            else:
                ego_state = self._ini_traj.state_at_time_step(time_step)
            ego_lon_s = convert_pos_curvilinear(ego_state, self._veh_config)[0]
            # if time_step in other_vehicle.states_lon:
            #    dist = other_vehicle.states_lon[time_step].s - ego_lon_s
            # else:
            #    continue
            try:
                dist = other_vehicle.get_lon_state(time_step, lane).s - ego_lon_s
            except:
                continue
            if 0 < dist < dist_pre:
                preceding_vehicle = other_vehicle
                dist_pre = dist
            elif 0 > dist > dist_post:
                following_vehicle = other_vehicle
                dist_post = dist
            else:
                continue
        return preceding_vehicle, following_vehicle

    def ConstrSpeedLimit(self, speed_limit):
        return [0, speed_limit]

    def ConstrAccNotAbruptly(self, a_abrupt):
        return [a_abrupt, np.inf]

    def ConstrCollisionFree(self):
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            if k in self._target_lanes:
                self._prec_veh, self._foll_veh = self._determine_related_veh(
                    k, self._target_lanes[k]
                )
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
                    self._lon_dis_constraints[index] = self._get_overlap(
                        self._lon_dis_constraints[index],
                        [
                            -np.inf,
                            self._prec_veh.rear_s(k, self._ego_vehicle.ref_path_lane)
                            - self._veh_config.wheelbase / 2
                            - self._veh_config.length / 2,
                        ],
                    )
            # discard the following vehicles since the scenario is not interactive
            # if self._foll_veh is not None:
            #     if k <= self._foll_veh.end_time:
            #         self._lon_dis_constraints[index] = self._get_overlap(self._lon_dis_constraints[index],
            #                                                              [self._foll_veh.front_s(k) +
            #                                                            self._veh_config.wheelbase/2,
            #                                                            np.inf])

    def ConstrCollisionFreeIntersection(self):
        """
        add collosion free constraints for intersections
        """
        for k in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            self._prec_veh, self._foll_veh = self._determine_related_veh_intersection(k)
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
                        self._lon_dis_constraints[index] = self._get_overlap(
                            self._lon_dis_constraints[index],
                            [
                                -np.inf,
                                s_min
                                - self._ego_vehicle.shape.length / 3
                                - self._veh_config.wheelbase / 2,
                            ],
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

    def ConstrInSameLane(self, time_step: int, prop_assignment: float):
        if time_step in self._target_vehicle.lanelet_assignment.keys():
            tar_veh_lanelet = self._target_vehicle.lanelet_assignment[time_step]
            try:
                tar_veh_lane = self._world_state.road_network.find_lane_by_lanelet(
                    list(tar_veh_lanelet)[0]
                )
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
                tar_veh_lane = [None]
                target_lane = [None]

        else:
            target_lane = [None]
        self._target_lanes[time_step] = list(set(target_lane))

    def ConstrInFrontOf(self, time_step: int, prop_assignment: float):
        # preventing KeyError
        if time_step > self._target_vehicle.end_time:
            return [-np.inf, np.inf]
        if prop_assignment > 0:
            rear_s = self._target_vehicle.rear_s(time_step)
            if rear_s:
                return [-np.inf, rear_s]
            else:
                return [-np.inf, np.inf]
        else:
            front_s = self._target_vehicle.front_s(time_step)
            if front_s:
                return [front_s, np.inf]
            else:
                return [-np.inf, np.inf]

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

    @staticmethod
    def lane_lateral_boundary(lane: Lane):
        pass

    @staticmethod
    def _get_overlap(interval1: list, interval2: list):
        return [max(interval1[0], interval2[0]), min(interval1[1], interval2[1])]

    def ConstrStopLine(self, time_step: int, prop_assignment: float):
        # TODO: check in qp planner
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
        return [-np.inf, upper_bound]

    def ConstrOnLaneletWithTypeIntersection(self, time_step: int):
        # TODO: nonconvex, check in qp planner
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
        lanelet_intersection = (
            self._rule_monitor.world.road_network.lanelet_network.find_lanelet_by_id(
                lanelet_intersection_id[0]
            )
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
            start_s - self._veh_config.length / 2 - self._veh_config.wheelbase / 2
        )
        return [-np.inf, upper_bound]

    def ConstrInIntersectionConflictAreaEgo(
        self, time_step: int, prop_assignment: float
    ):
        if prop_assignment == 0:
            return [-np.inf, np.inf]
        if prop_assignment < 0:
            if self._tc_obj.compliant_maneuver == Maneuver.BRAKE:
                front_constr = (
                    self.s_circle_center_front
                    - self._ego_vehicle.shape.length / 3
                    - self._veh_config.wheelbase / 2
                )
                return [-np.inf, front_constr]
            elif self._tc_obj.compliant_maneuver == Maneuver.KICKDOWN:
                rear_constr = (
                    self.s_circle_center_rear
                    + self._ego_vehicle.shape.length / 3
                    + self._veh_config.wheelbase / 2
                )
                return [rear_constr, np.inf]
            return [-np.inf, np.inf]
        else:
            return [-np.inf, np.inf]

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

        line_center_left_offset = shapely.offset_curve(
            line_center, ego_vehicle.shape.width / 2
        )
        conflict_center_left_offset = self.find_conflict_points(
            line_center_left_offset, conflict_area_shape
        )
        if conflict_center_left_offset is not None:
            s_center_left_offset = [
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *conflict_center_left_offset[0]
                )[0],
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *conflict_center_left_offset[1]
                )[0],
            ]
        line_center_right_offset = shapely.offset_curve(
            line_center, -ego_vehicle.shape.width / 2
        )
        conflict_center_right_offset = self.find_conflict_points(
            line_center_right_offset, conflict_area_shape
        )
        if conflict_center_right_offset is not None:
            s_center_right_offset = [
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *conflict_center_right_offset[0]
                )[0],
                ego_vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *conflict_center_right_offset[1]
                )[0],
            ]

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
