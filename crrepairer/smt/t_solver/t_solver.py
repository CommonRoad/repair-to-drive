import math
import time
import torch
from typing import List, Optional

from crrepairer.cut_off.tc import TC
from crrepairer.smt.t_solver.miqp_planner_repair import MIQPPlannerRepair
from crrepairer.smt.monitor_wrapper import STLRuleMonitor, PropositionNode
from crrepairer.utils.configuration import RepairerConfiguration

from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.planning.planning_problem import PlanningProblem

from commonroad_crime.utility.simulation import Maneuver

from crmonitor.predicates.position import PositionPredicates

tolerance = 1e-2  # tolerance for the gradient-based decision making


class TSolver:
    """
    T-solver for the SMT-based repairer.
    """

    def __init__(
        self,
        ego_vehicle: DynamicObstacle,
        rule_monitor: STLRuleMonitor,
        config: RepairerConfiguration,
    ):
        self._sel_prop = None
        self._prop_full = None
        self._rule_monitor = rule_monitor
        self._tc_obj = TC(ego_vehicle, rule_monitor)
        self._compliant_maneuvers = list()
        self._repairability = False
        self._planner = None
        self._miqp_planner = None
        self._planning_problem = config.planning_problem

        self.verbose = True
        self.config = config

        self._planner: Optional[MIQPPlannerRepair] = MIQPPlannerRepair(
            self._rule_monitor, self._tc_obj, self.config
        )

        self.tc_search_time = 0
        self.reach_set_time = 0
        self.opti_plan_time = 0

    @property
    def total_runtime(self):
        return self.tc_search_time + self.reach_set_time + self.opti_plan_time

    @property
    def tc_object(self):
        return self._tc_obj

    @property
    def compliant_maneuvers(self):
        return self._compliant_maneuvers

    def assign_proposition(
        self, propositions: List[PropositionNode], model: list, use_mpr_derivative: bool
    ):
        """
        Assigns propositions to the T-solver.
        """
        self._prop_full = propositions
        self._sel_prop = list()
        for prop in propositions:
            # if not the same value
            if prop is not None and prop.alphabet in model:
                if (prop.ttv_value < 0 and prop.alphabet[0] != "~") or (
                    prop.ttv_value > 0 and prop.alphabet[0] == "~"
                ):
                    self._sel_prop.append(prop)
                    if self.verbose:
                        print(
                            f"* \t<TSolver>: selected propositions: {prop.alphabet[-1]} {prop.name} = {prop.ttv_value}"
                        )
        self._compliant_maneuvers = self.set_compliant_maneuver(use_mpr_derivative)

    def set_compliant_maneuver(self, use_mpr_derivative: bool):
        """
        Set rule-compliant maneuvers based on the selected propositions.
        """
        assert self._sel_prop is not None, (
            "<T-Solver>: the atomic proposition needs to be "
            "assigned first for the T-solver"
        )
        compliant_maneuver = list()
        for prop_node in self._sel_prop:
            if (
                prop_node.name.startswith("previous")
                or "historically" in prop_node.name
            ):
                continue
            for predicate in prop_node.children:
                if not hasattr(predicate, "evaluator"):
                    continue
                # category of the predicate
                predicate_category = (
                    predicate.evaluator.predicate_name.__class__.__name__[:3]
                )
                if use_mpr_derivative:
                    # if prop_node.name == predicate.name:
                    # value at TV
                    if predicate.mpr_gradient is None:
                        return [Maneuver.BRAKE]
                    if torch.cuda.is_available():
                        grad_tensor = predicate.mpr_gradient[0]
                    else:
                        grad_tensor = predicate.mpr_gradient
                    # Check dtype and convert if necessary
                    if grad_tensor.dtype != torch.float64:
                        grad_tensor = grad_tensor.float()
                    else:
                        grad_tensor = grad_tensor.double()

                    # Detach, move to CPU, and convert to numpy
                    grad_list = grad_tensor.detach().cpu().numpy()
                    print(f"* predicate: {predicate.evaluator.predicate_name}")
                    if (
                        predicate_category == "Pos"
                        and predicate.evaluator.predicate_name
                        in [
                            PositionPredicates.KeepsSafeDistancePrec,
                            PositionPredicates.InFrontOf,
                            PositionPredicates.Precedes,
                            PositionPredicates.StopLineInFront,
                            PositionPredicates.InIntersectionConflictArea,
                        ]
                    ) or (predicate_category == "Vel"):
                        grad_a = grad_list[4]
                        print(f"* gradient list: {grad_list}")
                        print(f"* gradient towards lon input: {grad_a}")
                        print(
                            f"Multiplied value: {abs(predicate.latest_value * grad_a)}, Tolerance: {tolerance}"
                        )
                        if (
                            abs(predicate.latest_value * grad_a) <= tolerance
                        ):  # no decision can be made
                            compliant_maneuver += [Maneuver.BRAKE, Maneuver.KICKDOWN]
                            print(
                                "* \t<TSolver>: no decision can be made, both maneuvers are selected"
                            )
                        # positive to negative, robustness needs to be decreased (Delta rob < 0)
                        # negative to positive, robustness needs to be increased (Delta rob > 0)
                        elif -predicate.latest_value * grad_a > 0:  # delta v > 0
                            compliant_maneuver += [Maneuver.KICKDOWN]
                        else:  # delta v < 0
                            compliant_maneuver += [Maneuver.BRAKE]
                    elif predicate_category == "Pos":
                        grad_theta = grad_list[9]
                        print(f"* gradient list: {grad_list}")
                        print(f"* gradient towards lat input: {grad_theta}")
                        print(
                            f"Multiplied value: {abs(predicate.latest_value * grad_theta)}, Tolerance: {tolerance}"
                        )
                        if (
                            abs(predicate.latest_value * grad_theta) <= tolerance
                        ):  # no decision can be made
                            compliant_maneuver += [
                                Maneuver.STEERLEFT,
                                Maneuver.STEERRIGHT,
                            ]
                            print(
                                "* \t<TSolver>: no decision can be made, both maneuvers are selected"
                            )
                        elif (
                            -predicate.latest_value * grad_theta > 0
                        ):  # delta theta > 0
                            compliant_maneuver += [Maneuver.STEERLEFT]
                        else:  # delta theta < 0
                            compliant_maneuver += [Maneuver.STEERRIGHT]

                    elif predicate_category == "Acc":
                        compliant_maneuver += [Maneuver.CONSTANT]
                    else:
                        pass  # general predicate
                else:
                    print(
                        "* \t<TSolver>: Unfortunately, the model predictive robustness is"
                        " not really computed"
                    )
                    if (
                        predicate_category == "Pos"
                        and predicate.evaluator.predicate_name
                        in [
                            PositionPredicates.KeepsSafeDistancePrec,
                            PositionPredicates.InFrontOf,
                            PositionPredicates.Precedes,
                        ]
                    ):
                        compliant_maneuver += [Maneuver.BRAKE]  # , Maneuver.KICKDOWN]
                    elif (
                        predicate_category == "Pos"
                        and predicate.evaluator.predicate_name
                        in [PositionPredicates.StopLineInFront]
                    ):
                        compliant_maneuver += [Maneuver.BRAKE]
                    elif (
                        predicate_category == "Pos"
                        and predicate.evaluator.predicate_name
                        in [PositionPredicates.InIntersectionConflictArea]
                        and predicate.agent_placeholders == (0, 1)
                    ):
                        compliant_maneuver += [Maneuver.BRAKE]
                    elif (
                        predicate_category == "Pos"
                        and predicate.evaluator.predicate_name
                        in [PositionPredicates.InIntersectionConflictArea]
                        and predicate.agent_placeholders == (1, 0)
                    ):
                        # TODO: FIXME add maneuvers
                        pass
                        # compliant_maneuver += [Maneuver.STEERRIGHT, Maneuver.STEERLEFT]
                    elif (
                        predicate_category == "Pos"
                        and predicate.evaluator.predicate_name
                        in [PositionPredicates.OnLaneletWithTypeIntersection]
                    ):
                        compliant_maneuver += [Maneuver.BRAKE, Maneuver.KICKDOWN]
                    # elif predicate_category == "Pos":
                    #     # TODO: FIXME add maneuvers
                    #     pass
                    #     # compliant_maneuver += [Maneuver.STEERRIGHT, Maneuver.STEERLEFT]
                    elif predicate_category == "Vel":
                        compliant_maneuver += [Maneuver.BRAKE]
                    elif predicate_category == "Acc":
                        compliant_maneuver += [Maneuver.CONSTANT]
                    elif (
                        predicate_category == "Pos"
                        and predicate.evaluator.predicate_name
                        in [
                            PositionPredicates.MainCarriagewayRightLane,
                            PositionPredicates.InSameLane,
                        ]
                    ):
                        compliant_maneuver += [Maneuver.STEERRIGHT, Maneuver.STEERLEFT]
                    else:
                        pass  # general predicate
                        # raise ValueError('<T-Solver>: the category {} is not specified'
                        #                  .format(predicate_category))
        compliant_maneuver = list(set(compliant_maneuver))
        # Sort by the first letter, giving preference to "Brake" over "Steer/Turn"
        compliant_maneuver.sort(key=lambda m: (m.value[0] != "B", m.value))
        if not compliant_maneuver:
            print("* \t<TSolver>: no compliant maneuver is selected")
        else:
            string = "* \t<TSolver>: compliant maneuver /"
            for m in compliant_maneuver:
                string += m.value + "/"
            string += " is selected"
            print(string)
        return compliant_maneuver

    def search_tc(self):
        """
        Searches the time-to-compliance.
        """
        if self._compliant_maneuvers is None:
            return -math.inf  # marked as not repairable
        tc = self.tc_object.generate(self._compliant_maneuvers)
        return tc

    def _optimization_based_repair(self):
        """
        Initializes the qp planner and uses it for trajectory repairing.
        """
        self._planner.reset(tc_object=self.tc_object, rule_monitor=self._rule_monitor)
        suc = self._planner.construct_constraints(self._sel_prop, self._prop_full)
        if not suc:
            print("* \t<TSolver>: the constraints are not properly constructed")
            return
        print(f"* \t<TSolver>: MIQP planner is invoked")

        print(f"* \t<TSolver>: initialization time {self.reach_set_time:.3f}s")
        start_time = time.time()

        repaired_trajectory = self._planner.plan()
        # repaired_trajectory = self._miqp_planner.plan()

        self.reach_set_time = self._planner.reach_set_time
        self.opti_plan_time = self._planner.opti_plan_time
        print(f"* \t<TSolver>: solving time {time.time() - start_time:.3f}s")
        return repaired_trajectory

    def check(
        self, proposition: List[PropositionNode], model: list, use_mpr_derivative=False
    ) -> (bool, Trajectory):
        """
        Checks the T-consistency.
        """
        repaired_traj = None
        start_time = time.time()
        self.assign_proposition(proposition, model, use_mpr_derivative)
        if self.compliant_maneuvers is None:
            print("* \t<Tsolver>: tc = {}, tv = {}".format(-math.inf, -math.inf))
            return self._repairability, repaired_traj
        tc = self.search_tc()
        print(
            "* \t<Tsolver>: tc = {}, tv = {}".format(self._tc_obj.tc, self._tc_obj.tv)
        )
        self.tc_search_time += time.time() - start_time
        print(f"* \t<Tsolver>: run time {self.tc_search_time:.3f}s")
        if tc != -math.inf:
            repaired_traj = self._optimization_based_repair()
            if repaired_traj is not None:
                self._repairability = True
        return self._repairability, repaired_traj
