import os
import re
import time
from fractions import Fraction
from typing import List

from commonroad.scenario.obstacle import ObstacleType
from mpmath.libmp.libelefun import machin

from crrepairer.miqp_planner.configuration import (
    PlanningConfigurationVehicle,
)
from crrepairer.utils.constraints import (
    longitudinal_position_constraints,
    lateral_position_constraints,
    longitudinal_velocity_constraints,
)
from crrepairer.utils.configuration import RepairerConfiguration

from crmonitor.predicates.position import (
    PredInIntersectionConflictArea,
    PredStopLineInFront
)

from crmonitor.predicates.velocity import (
    PredLaneSpeedLimit,
    PredFovSpeedLimit,
    PredBrSpeedLimit,
    PredTypeSpeedLimit,
)
from crmonitor.predicates.acceleration import PredAbruptBreaking

from crrepairer.miqp_planner.constraints import LatConstraints, LonConstraints

from commonroad.scenario.trajectory import Trajectory

from commonroad.scenario.state import InitialState

from commonroad_reach_semantic.data_structure.config.semantic_configuration_builder import (
    SemanticConfigurationBuilder,
)
from commonroad_reach_semantic.data_structure.driving_corridor_extractor import (
    DrivingCorridorExtractor as DrivingCorridorExtractorSemantic,
)
from commonroad_reach_semantic.data_structure.reach.semantic_reach_interface import SemanticReachableSetInterface

from commonroad_reach.data_structure.reach.driving_corridor_extractor import DrivingCorridorExtractor
from commonroad_reach.data_structure.reach.reach_interface import ReachableSetInterface
import commonroad_reach_semantic.utility.visualization as util_visual_semantic
from commonroad_reach_semantic.data_structure.environment_model.semantic_model import (
    SemanticModel,
)
from commonroad_reach_semantic.data_structure.model_checking.spot_interface import (
    SpotInterface,
)
from commonroad_reach_semantic.data_structure.reach.semantic_labeling_reach_set_py import (
    PySemanticLabelingReachableSet,
)
from commonroad_reach_semantic.data_structure.rule.proposition import (
    Proposition
)
from commonroad_reach_semantic.data_structure.rule.traffic_rule_interface import (
    TrafficRuleInterface,
)

from crmonitor.common.vehicle import Vehicle
from crmonitor.predicates.position import (
    PredSafeDistPrec,
    PredInSameLane,
    PredInFrontOf,
)
from crrepairer.cut_off.tc import TC
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode


class RuleConstraintsReach:
    """
    Class for traffic rule constraints (manual definition)
    """

    def __init__(
        self,
        tc_object: TC,
        rule_monitor: STLRuleMonitor,
        veh_config: PlanningConfigurationVehicle,
        initial_trajectory: Trajectory,
        config_repair: RepairerConfiguration,
    ):
        # initialize the needed components
        self.repaired_rules = None
        self._tc_obj = tc_object
        self._rule_monitor = rule_monitor
        self._world_state = self._rule_monitor.world

        # ego vehicle
        self._ego_id = (
            self._rule_monitor.vehicle_id
        )  # if no target vehicle, the other_id stands for the ego
        self._ego_vehicle_cr = self._tc_obj.ego_vehicle
        self._ego_vehicle_world = self._world_state.vehicle_by_id(self._ego_id)
        self._ini_traj = initial_trajectory

        # other vehicle (rule-relevant)
        self._rule_to_other_id = self._rule_monitor.rule_to_other_id

        # todo: multiple target vehicles
        self._target_vehicle: Vehicle = self._world_state.vehicle_by_id(rule_monitor.other_id)

        # configuration
        self._veh_config = veh_config
        self._compliant_maneuver = None
        self._sel_prop_full = None
        self._prop_full = None
        self._nr_ts = None

        # initialize the commonroad-reach
        # we use the default path of the reach folder
        # todo: use params for path

        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_root = os.path.join(script_dir,
                                 "../../../../commonroad-reach-semantic")

        self.reach_config = SemanticConfigurationBuilder(
            path_root=path_root
        ).build_configuration(str(self._world_state.scenario.scenario_id))

        # update the time step and nr of computation
        self.reach_config.vehicle.ego.width = self._ego_vehicle_cr.obstacle_shape.width
        self.reach_config.vehicle.ego.length = self._ego_vehicle_cr.obstacle_shape.length
        self.reach_config.planning.dt = self._world_state.dt

        # update the path
        self.reach_config.general.path_scenarios = config_repair.general.path_scenarios
        self.reach_config.general.path_scenario = config_repair.general.path_scenario

        # path_scenarios = os.path.join(script_dir, '..', '..', '..', 'scenarios/')
        # self.reach_config.general.path_scenarios = path_scenarios
        # self.reach_config.general.path_scenario = os.path.join(path_scenarios,
        #                                                        f"{str(self._world_state.scenario.scenario_id)}.xml")

        # use the original scenario

        self.corridor = None

        self.reach_config.update()
        # # remove non-car obstacles
        # for obs in self.reach_config.scenario.obstacles:
        #     if obs.obstacle_type != ObstacleType.CAR:
        #         self.reach_config.scenario.remove_obstacle(obs)

        self.semantic_model = SemanticModel(self.reach_config)

        # update the rule interface
        self.rule_interface = TrafficRuleInterface(self.reach_config, self.semantic_model)

        # update params
        self.reach_config.vehicle.ego.t_react = 0.4
        self.reach_config.reachable_set.mode_computation = 8
        self.reach_config.vehicle.ego.a_lon_min = -10
        self.reach_config.vehicle.ego.v_max = 50
        if "R_IN4" in rule_monitor._rules:
            self.reach_config.vehicle.ego.a_lon_max = 4
            self.reach_config.vehicle.ego.v_max = 4
        else:
            self.reach_config.vehicle.other.a_lon_min = -10.5
            self.reach_config.vehicle.other.a_lon_max = 10.5
        if "R_G2" in rule_monitor._rules:
            self.reach_config.vehicle.other.a_lon_min = -2

        self.reach_config.planning.reference_point = "REAR"
        self.reach_config.vehicle.other.width = self._target_vehicle.shape.width
        self.reach_config.vehicle.other.length = self._target_vehicle.shape.length

        if self.reach_config.reachable_set.mode_computation in [7, 8]:
            self.reach_interface = SemanticReachableSetInterface(self.reach_config, self.semantic_model,
                                                                 self.rule_interface)
        else:
            # update the rule interface
            # initialize the reach interface
            self.reach_interface = ReachableSetInterface(self.reach_config)

            self.reach_interface._reach = PySemanticLabelingReachableSet(
                self.reach_config, self.semantic_model, self.rule_interface
            )

    def reset(self,
              start_time_step: int,
              tc_object: TC,
              rule_monitor: STLRuleMonitor,
              sel_proposition_full: List[PropositionNode],
              proposition_full: List[PropositionNode]
              ):
        if rule_monitor is not None:
            self._rule_monitor = rule_monitor
            self._rule_to_other_id = self._rule_monitor.rule_to_other_id

        if tc_object is not None:
            self._tc_obj = tc_object

            self._nr_ts = tc_object.N - tc_object.tc_time_step
            self.reach_config.planning.steps_computation = self._nr_ts

            self._compliant_maneuver = tc_object.compliant_maneuver

        if proposition_full is not None:
            self._prop_full = proposition_full

        if sel_proposition_full is not None:
            self._sel_prop_full = sel_proposition_full
            self.repaired_rules = []
            # add the repairing propositions
            for prop in self._prop_full:
                if prop is None or prop.name.startswith("previous") or "historically" in prop.name:
                    continue
                if PredSafeDistPrec.predicate_name in prop.name:
                    if prop.alphabet[0] == '~':
                        # change the sign
                        self.repaired_rules.append(
                            f'LTL G (!SafeDistance_V{self._rule_to_other_id[prop.source_rule]})')
                    else:
                        self.repaired_rules.append(
                            f'LTL G (SafeDistance_V{self._rule_to_other_id[prop.source_rule]})')
                elif PredInIntersectionConflictArea.predicate_name in prop.name:
                    semantic_prop = Proposition.in_conflict_with(self._rule_to_other_id[prop.source_rule])
                    if prop.name[5:6] != prop.name[7:8]:
                        pattern = r"once\[(.*?)\]"
                        time_interval = re.findall(pattern, prop.name)[0]
                        values = time_interval.split(",")
                        divided_values = [round(Fraction(value)/self.reach_config.planning.dt) for value in values]
                        time_steps = [self._tc_obj.tv_time_step - self._tc_obj.tc_time_step,
                                      self._tc_obj.N - self._tc_obj.tc_time_step]
                        time_steps[0] = max(time_steps[0] - divided_values[-1], 0)
                        time_interval_int = "..".join(str(value) for value in time_steps)
                        if prop.ttv_value == -float("inf"):
                            continue
                        if prop.ttv_value > 0:
                            # change the sign
                            semantic_prop = "!" + semantic_prop
                        else:
                            # fixme:
                            continue
                        self.repaired_rules.append(
                            'LTL G[' + time_interval_int + '](' + semantic_prop + ')')
                    else:
                        if prop.alphabet[0] == '~':
                            # change the sign
                            semantic_prop = "!" + semantic_prop
                        self.repaired_rules.append(
                            'LTL G(' + semantic_prop + ')')

                else:
                    if PredInSameLane.predicate_name in prop.name:
                        semantic_prop = Proposition.in_same_lane(self._rule_to_other_id[prop.source_rule])
                    elif PredInFrontOf.predicate_name in prop.name:
                        semantic_prop = Proposition.behind(self._rule_to_other_id[prop.source_rule])
                    elif PredStopLineInFront.predicate_name in prop.name:
                        semantic_prop = Proposition.behind_stop_line()
                    elif PredFovSpeedLimit.predicate_name in prop.name:
                        semantic_prop = Proposition.fov_speed_limit()
                    elif PredBrSpeedLimit.predicate_name in prop.name:
                        semantic_prop = Proposition.brake_speed_limit()
                    elif PredLaneSpeedLimit.predicate_name in prop.name:
                        semantic_prop = Proposition.lane_speed_limit()
                    elif PredTypeSpeedLimit.predicate_name in prop.name:
                        semantic_prop = Proposition.type_speed_limit()
                    else:
                        # for instance unnecessary_braking
                        semantic_prop = None
                    if semantic_prop:
                        if prop.alphabet[0] == '~':
                            # change the sign
                            semantic_prop = "!" + semantic_prop
                        time_steps = [self._tc_obj.tv_time_step - self._tc_obj.tc_time_step,
                                      self._tc_obj.N - self._tc_obj.tc_time_step]
                        time_interval_int = "..".join(str(value) for value in time_steps)

                        self.repaired_rules.append(
                            'LTL G[' + time_interval_int + '](' + semantic_prop + ')')
            print("* \t<TSolver>: activated rules", list(set(self.repaired_rules)))
            # self.reach_config.traffic_rule.activated_rules = list(set(repaired_rules))
            self.rule_interface.list_traffic_rules_activated = list(set(self.repaired_rules))
            for item in self.rule_interface.list_traffic_rules_activated:
                self.rule_interface._parse_traffic_rule(item, allow_abstract_rules=True)


    def update_reach_interface(
        self, vehicle_configuration: PlanningConfigurationVehicle
    ):
        # obtain the cut-off state
        cut_off_time_step = self._tc_obj.tc_time_step
        if cut_off_time_step == 0:
            cut_off_state = self._ego_vehicle_cr.initial_state
        else:
            cut_off_state = self._ini_traj.state_at_time_step(cut_off_time_step)

        assert cut_off_state.time_step == cut_off_time_step, (
            "the time step of the state_at_time_step "
            "doesn't match the corresponding state!"
        )

        # set the cut-off state as the initial state
        self.reach_config.planning_problem.initial_state = InitialState(
            position=cut_off_state.position,
            velocity=cut_off_state.velocity,
            orientation=cut_off_state.orientation,
            yaw_rate=0.0,
            slip_angle=0,
            time_step=cut_off_state.time_step,
            acceleration=cut_off_state.acceleration
        )

        # planning_problem has to be there!!!!
        self.reach_config.update(
            planning_problem=self.reach_config.planning_problem,
            scenario=self._tc_obj.scenario,  # with the target vehicle removed!!
            CLCS=self.reach_config.planning.CLCS,
        )

        #########################################################
        ##     update the velocity and acceleration rules      ##
        #########################################################
        self.reach_config.vehicle.ego.v_lon_min = 0  ## no driving backward
        # v_max = self.reach_config.vehicle.ego.v_max
        # a_min = self.reach_config.vehicle.ego.a_lon_min
        # for prop in self._sel_prop_full:
        #     # velocity limit
        #     for predicate in prop.children:
        #         if predicate.base_name in (
        #             PredFovSpeedLimit.predicate_name,
        #             PredBrSpeedLimit.predicate_name,
        #             PredTypeSpeedLimit.predicate_name,
        #             PredLaneSpeedLimit.predicate_name,
        #         ):
        #             speed_limit = predicate.evaluator.get_speed_limit(
        #                 self._world_state, self._tc_obj.tv_time_step, [self._ego_id]
        #             )
        #             if speed_limit:  # not None
        #                 if speed_limit < v_max:
        #                     v_max = speed_limit
        #         if predicate.base_name in [PredAbruptBreaking]:
        #             acc_min = predicate.evaluator.config["a_abrupt"]
        #             if acc_min > a_min:
        #                 a_min = acc_min
        #
        # self.reach_config.vehicle.ego.v_lon_max = v_max
        # self.reach_config.vehicle.ego.a_lon_min = a_min

        #########################################################
        # self.reach_interface = SemanticReachableSetInterface(self.reach_config, self.semantic_model,
        #                                                      self.rule_interface)
        self.reach_interface.reset(self.reach_config)
        # self.reach_interface.step_start = cut_off_state.time_step
        self.reach_interface.compute_reachable_sets(verbose=True)

        if self.reach_config.reachable_set.mode_computation in [5, 6]:
            self.spot_interface = SpotInterface(self.reach_interface, self.rule_interface)
            self.spot_interface.translate_ltl_formulas()
            self.spot_interface.translate_reachability_graph()
            self.spot_interface.check()

    def compute_semantic_reachable_set(self, vehicle_configuration, verbose=True):
        self.update_reach_interface(vehicle_configuration)

        # # * for debugging the reach semantic
        # # ==== plot computation results
        # if self.reach_config.reachable_set.mode_computation in [5, 6]:
        #     node_to_group = util_visual_semantic.groups_from_propositions(
        #         self.reach_interface._reach.labeler.reachable_set_to_propositions)
        # elif self.reach_config.reachable_set.mode_computation in [7, 8]:
        #     node_to_group = util_visual_semantic.groups_from_states(self.reach_interface._reach.reachable_set_to_label)
        # else:
        #     # no semantic information, so put all nodes in the same group
        #     node_to_group = defaultdict(lambda: 0)
        #
        # util_visual_semantic.plot_reach_graph(self.reach_interface, node_to_group=node_to_group)
        # util_visual_semantic.plot_scenario_with_regions(self.semantic_model, "CVLN")
        # util_visual_semantic.plot_scenario_with_reachable_sets(self.reach_interface, save_gif=True)

        # * for debugging the original reach
        #util_visual.plot_scenario_with_reachable_sets(self.reach_interface)

        if self.reach_config.traffic_rule.activated_rules and \
            self.reach_config.reachable_set.mode_computation in [5, 6]:
            dc_extractor = DrivingCorridorExtractorSemantic(self.spot_interface)
            dc_extractor.extract_corridors(search=True)
            self.corridor = dc_extractor.determine_optimal_corridor()
        else:
            dc_extractor = DrivingCorridorExtractor(self.reach_interface.reachable_set, self.reach_config)
            try:
                driving_corridors = dc_extractor.extract()
                self.corridor = driving_corridors[0]
            except Exception as e:
                print(f"Error in extracting the driving corridor: {e}")
                self.corridor = None


    def longitudinal_constraints(self, vehicle_configuration):
        # compute the driving corridor
        time_start = time.time()
        self.compute_semantic_reachable_set(vehicle_configuration)
        print(f"* \t<TSolver>: time for computing the reachable set {time.time()-time_start:.2f}")
        if self.corridor is None:
            raise Exception("the driving corridor is either not computed or empty")
        else:
            s_min, s_max = longitudinal_position_constraints(self.corridor)
            v_min, v_max = longitudinal_velocity_constraints(self.corridor)
        # if self.reach_config.traffic_rule.activated_rules:
        #     for prop in self._sel_prop_full:
        #         # velocity limit
        #         # fixme: adding stopping distance!!
        #         # for predicate in prop.children:
        #             # if predicate.base_name in [PredStopLineInFront.predicate_name]:
        #             #     for ts in range(self._tc_obj.tv_time_step - self._tc_obj.tc_time_step,
        #             #                     self._tc_obj.N - self._tc_obj.tc_time_step):
        #             #         s_max[ts] -= (predicate.evaluator.config["d_sl"])
        #
        #             #if predicate.base_name in [PredInIntersectionConflictArea.predicate_name]:
        #         for ts in range(self._tc_obj.tv_time_step - self._tc_obj.tc_time_step - 1,
        #                                 self._tc_obj.N - self._tc_obj.tc_time_step):
        #             s_max[ts] -= 0.1  #(self._veh_config.length/2)
        #             s_min[ts] -= 0.1
        c_tv_lon = LonConstraints.construct_constraints(
            s_min, s_max, s_min, s_max, v_min=v_min, v_max=v_max
        )
        return c_tv_lon

    def lateral_constraints(self, traj_lon, configuration_qp):
        traj_lon_positions = traj_lon.get_positions()[:, 0]
        # fixme: not the same interface as reach
        # lateral_driving_corridors = self.reach_interface.extract_driving_corridors(
        #     corridor_lon=self.corridor,
        #     list_p_lon=traj_lon_positions
        # )
        # lat_dc = list(lateral_driving_corridors)[0]
        d_min, d_max = lateral_position_constraints(
            self.corridor, self.corridor, traj_lon_positions, configuration_qp
        )
        c_tv_lat = LatConstraints.construct_constraints(d_min, d_max, d_min, d_max)
        return c_tv_lat
