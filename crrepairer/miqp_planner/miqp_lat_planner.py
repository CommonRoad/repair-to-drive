from typing import List, Optional
import numpy as np

from commonroad_clcs.util import (
    compute_pathlength_from_polyline,
)

from crrepairer.miqp_planner.gurobi_optimizer import GurobiSolver
from crrepairer.miqp_planner.miqp_constraints_manual import (
    LateralConstraint,
    TIConstraint,
)

from crrepairer.miqp_planner.trajectory import Trajectory, TrajPoint, TrajectoryType

from crrepairer.utils.configuration import RepairerConfiguration, VehicleConfiguration


class MIQPLatState(object):
    """
    Class representing a state <d,theta,kappa,kappa_dot> within the QPLatPlanner
    """

    def __init__(
        self,
        d: float,
        theta: float,
        kappa: float,
        kappa_dot: float,
        t=0.0,
        s=None,
        v=None,
        a=None,
        j=None,
        u_lon=None,
    ):
        self.d = d
        self.theta = theta
        self.kappa = kappa
        self.kappa_dot = kappa_dot
        self.t = t

        self.s = s
        self.v = v
        self.a = a
        self.j = j
        self.u_lon = u_lon


class MIQPLatRefState(object):
    """
    Class representing a state <s,v,theta,kappa> of QPLatPlanner reference (longitudinal profile and
    curvature/orientation of reference)
    """

    def __init__(
        self, s: float, v: float, a: float, j: float, theta: float, kappa: float
    ):
        self.s = s
        self.v = v
        self.a = a
        self.j = j
        self.theta = theta
        self.kappa = kappa


class MIQPLatReference(object):
    """
    Class representing a QPLatReference made up of a list of QPLatRefStates
    """

    def __init__(self, reference: List):
        self.reference = reference

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference: List):
        # check if reference is list of reference states
        assert isinstance(reference, list) and (
            isinstance(s, MIQPLatRefState) for s in reference
        )
        self._reference = reference

    def length(self) -> int:
        if isinstance(self.reference, list):
            return len(self.reference)
        else:
            return 1

    @classmethod
    def construct_from_lon_traj_and_reference(
        cls,
        lon_traj: Trajectory,
        reference: np.ndarray,
        vehicle_configuration: VehicleConfiguration,
    ) -> "MIQPLatReference":
        assert (
            isinstance(lon_traj, Trajectory)
            and lon_traj.coord_type is TrajectoryType.CARTESIAN
        ), "<MIQPLatReference>: Provided longitudinal trajectory is invalid or not in Frenet. traj = {}".format(
            lon_traj
        )
        assert np.isclose(
            np.sum(lon_traj.get_positions()[:, 1]), 0.0
        ), "<MIQPLatReference>: Provided longitudinal trajectory containts lateral information != 0. d = {}".format(
            lon_traj.get_positions()[:, 1]
        )
        assert (
            isinstance(reference, np.ndarray)
            and reference.ndim == 2
            and len(reference) > 1
            and len(reference[0, :]) == 2
        ), "<MIQPLatReference>: Provided reference is not valid. reference = {}".format(
            reference
        )

        # compute orientation, curvature and pathlength of reference
        # ref_orientation = compute_orientation_from_polyline(reference)
        # ref_curvature = compute_curvature_from_polyline(reference)
        ref_pathlength = compute_pathlength_from_polyline(reference)

        # get s coordinates of longitudinal motion for interpolation of theta and kappa of reference
        s = lon_traj.get_positions()[:, 0]
        v = lon_traj.get_velocities()
        a = lon_traj.get_accelerations()
        j = lon_traj.get_jerk()
        # check if numerical errors have happened in lon trajectory
        for i in range(0, len(a)):
            if np.greater(a[i], vehicle_configuration.qp_veh_config.a_max_x):
                a[i] = vehicle_configuration.qp_veh_config.a_max_x
            if np.greater(vehicle_configuration.qp_veh_config.a_min_x, a[i]):
                a[i] = vehicle_configuration.qp_veh_config.a_min_x

        assert np.greater_equal(
            np.max(ref_pathlength), np.max(s)
        ), "<QPLatReference>: Provided reference is not long enough for interpolation! ref = {}, traj = {}".format(
            np.max(ref_pathlength), np.max(s)
        )

        # interpolate curvature at s positions of trajectory
        # curvature_interpolated = np.interp(s, ref_pathlength, ref_curvature)
        curvature_interpolated = np.interp(
            s,
            vehicle_configuration.qp_veh_config.ref_pos,
            vehicle_configuration.qp_veh_config.ref_curv,
        )

        # interpolate orientation at s positions of trajectory
        # orientation_interpolated = np.interp(s, ref_pathlength, ref_orientation)
        orientation_interpolated = np.interp(
            s,
            vehicle_configuration.qp_veh_config.ref_pos,
            vehicle_configuration.qp_veh_config.ref_theta,
        )
        assert (
            len(curvature_interpolated) == len(orientation_interpolated) == len(s)
        ), "<QPLatReference>: interpolation failed!"

        # create QPLat reference
        states = list()
        for i in range(len(s)):
            states.append(
                MIQPLatRefState(
                    s[i],
                    v[i],
                    a[i],
                    j[i],
                    orientation_interpolated[i],
                    curvature_interpolated[i],
                )
            )

        return MIQPLatReference(states)


class MIQPLatPlanner:
    def __init__(
        self,
        config: RepairerConfiguration,
    ):
        self.time_horizon = config.miqp_planner.horizon
        self.N = config.miqp_planner.N_p
        self.dt = config.scenario.dt

        # initialize from the configuration
        self.config: Optional[RepairerConfiguration] = None
        self.reset(config)

        # number of x <d, theta, kappa, kappa dot>
        self._n = 4
        # number of u <kappa dot dot>
        self._m = 1

        # slack variable
        self._slack_pos = config.miqp_planner.slack_lat
        self._n_s = 2 if self._slack_pos else 0

        # wheelbase length
        self._wb_length = self.config.vehicle.qp_veh_config.wheelbase
        self._x_init_lat = None
        self._x_ref_lat = None

        self.config = config
        self.weight = config.miqp_planner.weight_lat

        self.d_reference = None

        self.solver = GurobiSolver()

        self.theta_r = list()

        self.dynamic_matrix_list = None
        self.lat_dis_cons_matrix = None
        self.kappa_lim = None

    def reset(
        self,
        config: RepairerConfiguration = None,
        x_init_lat: MIQPLatState = None,
        x_ref_lat: MIQPLatReference = None,
        nr_steps: int = None,
        horizon: float = None,
        d_reference=None,
    ):
        # set updated config
        if config is not None:
            self.config = config
        else:
            assert (
                self.config is not None
            ), "<MIQP LONG PLANNER.reset(). No Configuration object provided>"
        if x_init_lat is not None:
            self._x_init_lat = x_init_lat
        if x_ref_lat is not None:
            self._x_ref_lat = x_ref_lat
            self.theta_r = list()
            for i in range(self.N):
                self.theta_r.append(self._x_ref_lat.reference[i].theta)

            (
                self.dynamic_matrix_list,
                self.lat_dis_cons_matrix,
                self.kappa_lim,
            ) = self._init_dynamic_matrices()

        if d_reference is not None:
            self.d_reference = d_reference

        if nr_steps is not None:
            self.N = nr_steps

            if self.d_reference is None:
                self.d_reference = np.zeros(self.N + 1)

        if horizon is not None:
            self.time_horizon = horizon

    def plan(
        self, lateral_constraints: LateralConstraint, ti_constraints: TIConstraint
    ):
        """Plan the lateral movement based on the constraints and longitudinal one."""
        # initialize state and control variables and add time-invariant constraints
        self._init_state_var(ti_constraints)
        self._init_control_var(ti_constraints)

        # add lateral dynamic constraints
        init_state = np.array(
            [
                self._x_init_lat.d,
                self._x_init_lat.theta,
                self._x_init_lat.kappa,
                self._x_init_lat.kappa_dot,
            ]
        ).transpose()
        self.solver.add_lat_dynamic_cons(
            self.dynamic_matrix_list,
            init_state,
            self.theta_r,
        )

        if self._slack_pos:
            self._init_slack_var(ti_constraints)

        # add time-variant constraint
        self.solver.add_lat_dis_cons(
            self.lat_dis_cons_matrix,
            self._x_ref_lat,
            lateral_constraints.d_min,
            lateral_constraints.d_max,
        )
        self.solver.add_kappa_limit(self.kappa_lim)

        # cost function
        self.solver.costfunc_lat(
            self._x_ref_lat,
            self.weight,
            d_reference=self.d_reference,
            lat_cons=lateral_constraints,
        )
        self.solver.solve()

        # get solution
        try:
            print("slack variable: ", self.solver.get_slack_var())
            trajectory = self.create_output_trajectory(lateral_constraints.long_traj)
        except:
            # Compute an Irreducible Inconsistent Subsystem (IIS)
            self.solver.model.computeIIS()
            self.solver.model.write("model_lat.ilp")
            return None  # fixme: better handling needed, add warning
        return trajectory

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

    def _init_state_var(self, ti_constraints: TIConstraint):
        """Initializes the state variables and adds time-invariant constraints"""
        x_shape = (self._n, self.N + 1)
        x = np.empty(x_shape, dtype=object)
        c_ti_lb = [
            ti_constraints.d_min,
            ti_constraints.theta_min,
            ti_constraints.kappa_min,
            ti_constraints.kappa_dot_min,
        ]
        c_ti_ub = [
            ti_constraints.d_max,
            ti_constraints.theta_max,
            ti_constraints.kappa_max,
            ti_constraints.kappa_dot_max,
        ]
        self.solver.add_lat_state_var(
            x,
            x_shape,
            c_ti_lb,
            c_ti_ub,
        )

    def _init_control_var(self, ti_constraints: TIConstraint):
        """Initialize control variables."""
        u_shape = (self.N,)
        u = np.empty(u_shape, dtype=object)
        self.solver.add_lat_control_var(
            u,
            u_shape,
            ti_constraints.kappa_dot_dot_min,
            ti_constraints.kappa_dot_dot_max,
        )

    def _init_dynamic_matrices(self):
        dynamic_matrix_list = list()
        lat_dis_cons_matrix = list()

        kappa_lim = list()

        for i in range(self.N):
            v = self._x_ref_lat.reference[i].v
            a = self._x_ref_lat.reference[i].a
            theta = self._x_ref_lat.reference[i].theta

            # x = Ax+Bu+Dz
            A = np.array(
                [
                    [
                        1,
                        self.dt * v,
                        (self.dt**2) * 0.5 * (v**2),
                        (self.dt**3) / 6 * (v**2),
                    ],
                    [0, 1, self.dt * v, (self.dt**2) * 0.5 * v],
                    [0, 0, 1, self.dt],
                    [0, 0, 0, 1],
                ]
            )
            B = np.array(
                [
                    [(self.dt**4) / 24 * (v**2)],
                    [(self.dt**3) / 6 * v],
                    [(self.dt**2) * 0.5],
                    [self.dt],
                ]
            )
            # disturbances on input
            D = np.array([-self.dt * v, 0, 0, 0]).reshape([-1, 1])
            dynamic_matrix_list.append({"A": A, "B": B, "D": D})

            # selection matrix for output
            S = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
            C = np.array(
                [
                    [1, 0, 0, 0],
                    [1, 0.5 * self._wb_length, 0, 0],
                    [1, self._wb_length, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            # disturbances on output
            E = np.transpose(
                np.array([0, -0.5 * self._wb_length, -self._wb_length, 0, 0])
            )
            lat_dis_cons_matrix.append({"S": S, "C": C, "E": E})

            kappa_lim_k = min(
                np.sqrt(self.config.vehicle.qp_veh_config.a_max**2 - a**2)
                / (np.max([v, 0.5]) ** 2),
                self.config.vehicle.kappa_max,
            )
            kappa_lim.append(kappa_lim_k)
        # lateral_constraints.kappa_lim = np.array(kappa_lim)
        return dynamic_matrix_list, lat_dis_cons_matrix, np.array(kappa_lim)

    def create_output_trajectory(self, long_traj: Trajectory):
        var_x = self.solver.get_var_x()
        control_u = self.solver.get_control_u()
        long_traj_states = long_traj.states
        traj = list()
        # add initial state
        traj.append(
            TrajPoint(
                self._x_init_lat.t,
                self._x_init_lat.s,
                self._x_init_lat.d,
                self._x_init_lat.theta,
                self._x_init_lat.v,
                self._x_init_lat.a,
                j=self._x_init_lat.j,
            )
        )
        for k in range(self.N):
            traj.append(
                TrajPoint(
                    t=self._x_init_lat.t + self.dt * (k + 1),
                    x=long_traj_states[k + 1].position[0],
                    y=var_x[0, k + 1],
                    theta=var_x[1, k + 1],
                    v=long_traj_states[k + 1].v,
                    a=long_traj_states[k + 1].a,
                    kappa=var_x[2, k + 1],
                    j=long_traj_states[k + 1].j,
                    kappa_dot=var_x[3, k + 1],
                    lane=-1,
                )
            )
        traj = Trajectory(traj, TrajectoryType.CARTESIAN)
        traj._u_lon = long_traj.u_lon
        traj._u_lat = control_u
        return traj
