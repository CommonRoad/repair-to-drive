import time

from crrepairer.smt.t_solver.rule_constraints_reach import RuleConstraintsReach
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.cut_off.tc import TC
from crrepairer.miqp_planner.configuration import PlanningConfigurationVehicle
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.constraints import (
    longitudinal_position_constraints,
    lateral_position_constraints,
    longitudinal_velocity_constraints,
)

from crmonitor.common.vehicle import Vehicle

from commonroad.scenario.trajectory import Trajectory

from crrepairer.miqp_planner.miqp_long_planner import LongitudinalConstraint
from crrepairer.miqp_planner.miqp_constraints_manual import PredicateConstraint, LateralConstraint

from crrepairer.miqp_planner.trajectory import Trajectory as QPTrajectory


class RuleConstraintMIQPReach(RuleConstraintsReach):
    def __init__(self,
                 tc_object: TC,
                 rule_monitor: STLRuleMonitor,
                 veh_config: PlanningConfigurationVehicle,
                 initial_trajectory: Trajectory,
                 config_repair: RepairerConfiguration
                 ):
        super().__init__(tc_object, rule_monitor, veh_config, initial_trajectory, config_repair)
        self.longitudinal_constraints = None
        self.lateral_constraints = None
        self.safe_distance_modes = None

    @property
    def target_vehicle(self) -> Vehicle:
        return self._target_vehicle

    def construct_longitudinal_constraints(self, vehicle_configuration, tc_time_step):
        """Construct the longitudinal constraints from the driving corridor."""
        # all set as false as it is considered in the reachable set computation
        self.compute_semantic_reachable_set(vehicle_configuration)

        self.safe_distance_modes = [
            False for _ in range(self._tc_obj.N - self._tc_obj.tc_time_step + 1)
        ]
        self.longitudinal_constraints = LongitudinalConstraint(
            self._sel_prop_full
        )
        self.longitudinal_constraints.tc = tc_time_step

        # compute the reachable set
        time_start = time.time()
        print(f"* \t<TSolver>: time for computing the reachable set {time.time()-time_start:.2f}")

        if self.corridor is None:
            print("the driving corridor is either not computed or empty")
            return
        else:
            s_min, s_max = longitudinal_position_constraints(self.corridor, FULL=True)
            v_min, v_max = longitudinal_velocity_constraints(self.corridor, FULL=True)

        self.longitudinal_constraints.rule_constraints[
            "reach_position"
        ] = PredicateConstraint(
            decision_variable=False,
            num_decision_variables=0,
            constraint_state=0,
            constraint_name="position",
            start_time_step=self._tc_obj.tc_time_step,
            end_time_step=self._tc_obj.N,
        )
        self.longitudinal_constraints.rule_constraints[
            "reach_velocity"
        ] = PredicateConstraint(
            decision_variable=False,
            num_decision_variables=0,
            constraint_state=1,
            constraint_name="velocity",
            start_time_step=self._tc_obj.tc_time_step,
            end_time_step=self._tc_obj.N,
        )
        for time_step in range(self._tc_obj.tc_time_step, self._tc_obj.N + 1):
            index = time_step - self._tc_obj.tc_time_step
            self._get_overlap("reach_position", s_max[index], s_min[index], time_step)
            self._get_overlap("reach_velocity", v_max[index], v_min[index], time_step)

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

    def create_d_constraints(self, long_traj: QPTrajectory, configuration_qp):
        self.lateral_constraints = LateralConstraint(self._sel_prop_full)
        self.lateral_constraints.long_traj = long_traj

        # create the d constraints from the driving corridor
        traj_lon_positions = long_traj.get_positions()[:, 0]
        d_min, d_max = lateral_position_constraints(
            self.corridor, self.corridor, traj_lon_positions, configuration_qp
        )
        self.lateral_constraints.d_min = d_min
        self.lateral_constraints.d_max = d_max

