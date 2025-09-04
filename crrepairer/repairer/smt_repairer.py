from abc import ABC
import math
import time

from commonroad.scenario.scenario import DynamicObstacle
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.obstacle import TrajectoryPrediction, ObstacleType
from commonroad.planning.planning_problem import PlanningProblem

from crrepairer.repairer.base import TrajectoryRepair
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.smt.sat_solver.sat_solver import SATSolver
from crrepairer.smt.t_solver.t_solver import TSolver

from z3 import sat
from enum import Enum


class RepairingRule(Enum):
    COLLISION_FREE = "collision free"
    R_G1 = "R_G1"
    R_G2 = "R_G2"
    R_G3 = "R_G3"


class SMTTrajectoryRepairer(TrajectoryRepair, ABC):
    def __init__(
        self,
        rule_monitor: STLRuleMonitor,
        ego_vehicle: DynamicObstacle,
        config: RepairerConfiguration,
    ):
        super().__init__(ego_vehicle.prediction.trajectory)
        self.rule_monitor = rule_monitor
        self._model = None
        self._tc = -math.inf
        self._tv = -math.inf
        # initialize Solvers for SMT paradigm

        self.sat_solver = SATSolver(self.rule_monitor, config)

        self.t_solver = TSolver(ego_vehicle, self.rule_monitor, config)
        self.config = config

        # Runtime tracking variables
        self.sat_reasoning_time = 0

    @property
    def tv(self):
        return self._tv

    @property
    def tc(self):
        return self._tc

    @property
    def target_vehicle(self):
        return self.config.scenario.obstacle_by_id(self.rule_monitor.other_id)

    @property
    def model(self):
        """
        SAT models
        """
        return self._model

    def repair(self, check_flag=True, *args, **kwargs):
        self._tv = self.rule_monitor.tv_time_step
        if self._tv == -math.inf:
            return None
        nr = 1
        print("******** Trajectory Repairing starts! ********")
        start_time = time.time()
        while self.sat_solver.solve() == sat:
            print("* {}. iteration...".format(nr))
            if self.rule_monitor.proposition_nodes is None:
                return None
            sat_start_time = time.time()
            select_proposition, self._model = self.sat_solver.model()
            self.sat_reasoning_time += time.time() - sat_start_time
            print(
                "* \t<SATSolver>: SAT reasoning time: {:.3f}s".format(
                    self.sat_reasoning_time
                )
            )
            repairability, repaired_traj = self.t_solver.check(
                select_proposition,
                list(self._model),
                use_mpr_derivative=self.config.repair.use_mpr_derivative,
            )
            self._tc = self.t_solver.tc_object.tc_time_step
            if repairability and repaired_traj is not None:
                print(f"----- Computation Time: {time.time() - start_time:.3f}s -----")
                print("*****  Successfully Repaired! •ᴗ•  *****")
                print(
                    f"----- Time details ----- \n***** SAT: {self.sat_reasoning_time:.6f}s"
                    f"\n***** TC search: {self.t_solver.tc_search_time:.3f}s"
                    f"\n***** Reachset computation: {self.t_solver.reach_set_time:.3f}s"
                    f"\n***** Optimization: {self.t_solver.opti_plan_time:.3f}s"
                    f"\n***** Total: {self.sat_reasoning_time + self.t_solver.total_runtime:.3f}s"
                )
                return repaired_traj
                # tv, _ = self.t_solver.tc_object.calc_tv_updated(
                #     repaired_traj.state_list, int(self._tc)
                # )
                # if tv == math.inf or not check_flag:
                #     print("*****  Successfully Repaired! •ᴗ•  *****")
                #     return repaired_traj
                # else:
                #     print("*** Reparable but Solver Failed ಠ_ಠ  ***")
            self.sat_solver.update_formula()
            nr += 1
        print("*******   Repairing Failed ಠ_ಠ   *******")
        return None

    @staticmethod
    def convert_traj_to_ego_vehicle(
        shape, initial_state, cr_trajectory: Trajectory, vehicle_id: int = 0
    ) -> DynamicObstacle:
        """
        Converts trajectory object to CommonRoad obstacle with specified width and length
        :param shape: The vehicle shape
        :param initial_state: The initial state of the ego vehicle
        :param vehicle_id: ID of ego vehicle
        :return: The CommonRoad DynamicObstacle object containing the current trajectory
        """
        # get trajectory
        pred = TrajectoryPrediction(cr_trajectory, shape)

        # create new object
        ego = DynamicObstacle(
            obstacle_id=vehicle_id,
            obstacle_type=ObstacleType.CAR,
            prediction=pred,
            obstacle_shape=shape,
            initial_state=initial_state,
        )
        return ego
