from typing import List, Optional
import numpy as np
from decimal import Decimal

from crrepairer.miqp_planner.gurobi_optimizer import GurobiSolver
from crrepairer.miqp_planner.miqp_constraints_manual import (
    LongitudinalConstraint,
    TIConstraint,
)

from crrepairer.utils.configuration import RepairerConfiguration
from crmonitor.common.vehicle import Vehicle
from crrepairer.miqp_planner.trajectory import Trajectory, TrajPoint, TrajectoryType


class MIQPLongState(object):
    def __init__(self, s: float, v: float, a: float, j: float, t=0.0):
        self.s = s
        self.v = v
        self.a = a
        self.j = j
        self.t = t


class MIQPLongReference(object):
    def __init__(self, state):
        self.reference = state

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, state):
        # check if state is single state or list of states
        assert isinstance(state, MIQPLongState) or (
            isinstance(state, list) and (isinstance(s, MIQPLongState) for s in state)
        )
        self._reference = state

    def length(self) -> int:
        if isinstance(self.reference, list):
            return len(self.reference)
        else:
            return 1


class MIQPLongPlanner:
    def __init__(
        self,
        config: RepairerConfiguration,
    ):
        # basic configuration
        self.time_horizon = config.miqp_planner.horizon
        self.N = config.miqp_planner.N_p
        self.dt = config.scenario.dt
        self.scenario = config.scenario
        self.vehicle_configuration = config.vehicle

        # construct the initial state
        self.initial_state: Optional[TrajPoint] = None
        self.s0: Optional[MIQPLongState] = None
        self.init_state_long: Optional[np.array] = None

        self.config: Optional[RepairerConfiguration] = None
        self.reset(config)

        self.weight = config.miqp_planner.weight_long

        # number of x
        self._n = 4
        # number of u
        self._m = 1

        # slack variable
        self._slack_pos = config.miqp_planner.slack_long
        self._n_s = 2 if self._slack_pos else 0

        # Dynamic matrix and initial state
        A = np.array(
            [
                [1, self.dt, (self.dt**2.0) / 2.0, (self.dt**3.0) / 6.0],
                [0, 1.0, self.dt, (self.dt**2.0) / 2.0],
                [0, 0, 1.0, self.dt],
                [0, 0, 0, 1],
            ]
        )
        B = np.array(
            [
                [(self.dt**4.0) / 24.0],
                [(self.dt**3.0) / 6.0],
                [(self.dt**2.0) / 2.0],
                [self.dt],
            ]
        )
        D = np.array([0, 0, 0, 0]).reshape([-1, 1])

        self.dynamic_matrix_list = [{"A": A, "B": B, "D": D}]

        self._velocity_samples = None

        # initialize solver
        self.solver = GurobiSolver()

    def reset(
        self,
        config: RepairerConfiguration = None,
        initial_state: TrajPoint = None,
        nr_steps: int = None,
        horizon: float = None,
    ):
        """resets the planner"""
        # set updated config
        if config is not None:
            self.config = config
        else:
            assert (
                self.config is not None
            ), "<MIQP LONG PLANNER.reset(). No Configuration object provided>"

        if initial_state is not None:
            self.initial_state = initial_state
            self.s0 = MIQPLongState(
                self.initial_state.position[0],
                self.initial_state.v,
                self.initial_state.a,
                self.initial_state.j,
            )
            self.init_state_long = np.array(
                [self.s0.s, self.s0.v, self.s0.a, self.s0.j]
            ).transpose()

        if horizon is not None:
            if Decimal(str(horizon)) % Decimal(str(self.scenario.dt)) != Decimal("0.0"):
                raise ValueError(
                    "<MIQPPlanner>: the given time step {} is inappropriate,"
                    "since time horizon is {}.".format(
                        self.scenario.dt, config.miqp_planner.horizon
                    )
                )
            self.time_horizon = horizon

        if nr_steps is not None:
            self.N = nr_steps
            self.dynamic_matrix_list = [self.dynamic_matrix_list[0]] * self.N

    def plan(
        self,
        long_ref: MIQPLongReference,
        ti_constraints: TIConstraint,
        long_constraints: LongitudinalConstraint,
        safe_distance_modes: List[bool],
        pre_vehicle: Vehicle,
    ):
        # initial state and control variables in solver and add time-invariant constraints
        self._init_state_var(ti_constraints)
        self._init_control_var(ti_constraints)

        # add longitudinal dynamic constraints and constraints for initial state
        self.solver.add_long_dynamic_cons(
            self.dynamic_matrix_list, self.init_state_long
        )

        if self._slack_pos:
            self._init_slack_var(ti_constraints)

        # add rule constraints in solver
        self.solver.add_rule_cons(long_constraints.rule_constraints)
        # add collision free constraints in solver
        # self.solver.add_collision_free_cons(long_constraints.collision_free_constraints)
        # set velocity sample for approximate safe distances
        self._velocity_samples = np.linspace(0, ti_constraints.v_x_max, 10)
        # add safe distance constraints
        if any(safe_distance_modes):
            # only adding the distance when the safe distance mode is activated for some time steps
            self.solver.add_safe_distance_cons(
                safe_distance_modes,
                pre_vehicle,
                self._velocity_samples,
                ti_constraints,
                long_constraints.tc,
            )
        # cost function
        self.solver.costfunc_long(long_ref, self.weight)
        self.solver.solve()

        try:
            # extract solution
            print("slack variable: ", self.solver.get_slack_var())
            trajectory = self.create_output_trajectory()
        except:
            # Compute an Irreducible Inconsistent Subsystem (IIS)
            self.solver.model.computeIIS()
            self.solver.model.write("model_long.ilp")
            return None  # fixme: better handling needed, add warning

        return trajectory

    def _init_state_var(self, ti_constraints: TIConstraint):
        """Initializes the state variables and adds time-invariant constraints"""
        x_shape = (self._n, self.N + 1)
        x = np.empty(x_shape, dtype=object)
        c_ti_lb = [
            ti_constraints.x_min,
            ti_constraints.v_x_min,
            ti_constraints.a_x_min,
            ti_constraints.j_x_min,
        ]
        c_ti_ub = [
            ti_constraints.x_max,
            ti_constraints.v_x_max,
            ti_constraints.a_x_max,
            ti_constraints.j_x_max,
        ]
        # add longitudinal state variables
        self.solver.add_long_state_var(
            x,
            x_shape,
            c_ti_lb,
            c_ti_ub,
        )

    def _init_control_var(self, ti_constraints: TIConstraint):
        """Initializes the input variables and adds time-invariant constraints"""
        u_shape = (self.N,)
        u = np.empty(u_shape, dtype=object)
        self.solver.add_long_control_var(
            u,
            u_shape,
            ti_constraints.j_dot_x_min,
            ti_constraints.j_dot_x_max,
        )

    def _init_slack_var(self, ti_constraints: TIConstraint):
        """Initializes slack variables and adds time-invariant constraints"""
        slack_shape = (self._n_s,)
        slack = np.empty(slack_shape, dtype=object)
        self.solver.add_slack_var(
            slack,
            slack_shape,
            ti_constraints.slack_min,
            ti_constraints.slack_max,
        )

    def create_output_trajectory(self):
        """creates the output trajectory"""
        # extract the solution
        var_x = self.solver.get_var_x()
        var_u = self.solver.get_control_u()

        # generate the trajectory
        traj = list()
        # add initial state
        traj.append(
            TrajPoint(
                self.initial_state.t,
                self.initial_state.position[0],
                0,
                0,
                self.initial_state.v,
                self.initial_state.a,
                j=self.initial_state.j,
            )
        )
        for k in range(self.N):
            traj.append(
                TrajPoint(
                    self.initial_state.t + self.dt * (k + 1),
                    var_x[0, k + 1],
                    0,
                    0,
                    var_x[1, k + 1] if var_x[1, k + 1] >= 0.0 else 0.0,
                    var_x[2, k + 1],
                    j=var_x[3, k + 1],
                )
            )
        traj = Trajectory(traj, TrajectoryType.CARTESIAN)
        traj._u_lon = var_u
        return traj
