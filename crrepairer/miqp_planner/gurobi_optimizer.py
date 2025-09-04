import math
from typing import List, Union, Dict, Tuple

from gurobipy import Model as Gmodel
from gurobipy import QuadExpr, LinExpr, GRB

import numpy as np

from crrepairer.miqp_planner.miqp_constraints_manual import (
    LongitudinalConstraint,
    PredicateConstraint,
    CollisionFreeConstraint,
    TIConstraint,
    LateralConstraint,
)
from crrepairer.miqp_planner.utils import (
    calculate_safe_distance,
    derivative_safe_distance,
)
from crmonitor.predicates.base import BasePredicateEvaluator
from crmonitor.predicates.position import PredInSameLane
from crmonitor.common.vehicle import Vehicle


class GurobiSolver:
    def __init__(self):
        self.u_shape = None
        self.u = None
        self.x_shape = None
        self.x = None
        self.slack = None
        self.constraint_obj = None
        self.delta = {}
        self.model = Gmodel()

    def add_long_state_var(self, x: np.ndarray, x_shape: Tuple, x_lb: List, x_ub: List):
        """Add longitudinal states and time-invariant constraints (lower and upper bounds)"""
        # TODO: lower and upper bounds for the initial state
        self.x = x
        self.x_shape = x_shape
        for i in range(self.x_shape[0]):  # for each state variable
            for j in range(self.x_shape[1]):  # for each time step
                self.x[i, j] = self.add_var(
                    "continuous", "x_long_{}_{}".format(i, j), x_lb[i], x_ub[i]
                )

    def add_long_control_var(
        self, u: np.ndarray, u_shape: Tuple, u_lb: float, u_ub: float
    ):
        """Add longitudinal control and time-invariant constraints (lower and upper bounds)"""
        self.u = u
        self.u_shape = u_shape
        for i in range(self.u_shape[0]):  # for each time step
            self.u[i] = self.add_var("continuous", "u_long_{}".format(i), u_lb, u_ub)

    def add_slack_var(
        self, slack: np.ndarray, slack_shape: Tuple, slack_lb: float, slack_ub: float
    ):
        """Add slack variables and boundaries"""
        self.slack = slack
        self.slack_shape = slack_shape
        for i in range(self.slack_shape[0]):
            self.slack[i] = self.add_var(
                "continuous", "slack_{}".format(i), slack_lb, slack_ub
            )

    def add_lat_state_var(self, x: np.ndarray, x_shape: Tuple, x_lb: List, x_ub: List):
        self.x = x
        self.x_shape = x_shape
        for i in range(self.x_shape[0]):
            for j in range(self.x_shape[1]):
                self.x[i, j] = self.add_var(
                    "continuous", "x_lat_{}_{}".format(i, j), x_lb[i], x_ub[i]
                )

    def add_lat_control_var(
        self, u: np.ndarray, u_shape: Tuple, u_lb: float, u_ub: float
    ):
        self.u = u
        self.u_shape = u_shape
        for i in range(self.u_shape[0]):
            self.u[i] = self.add_var("continuous", "u_lat_{}".format(i), u_lb, u_ub)

    def add_var(self, typeofvar, name, lb=0.0, ub=0.0):
        """Add continuous or binary variable in gurobi model"""
        if typeofvar == "continuous":
            return self.model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=name)
        if typeofvar == "binary":
            return self.model.addVar(vtype=GRB.BINARY, name=name)

    def add_long_dynamic_cons(self, dynamic_matrix_list, init_state):
        """
        Add the longitudinal dynamic constraints
        x_(k+1) = A * x_k + B * u_k
        disturbance term z = 0
        """
        z = np.zeros(self.x_shape[1])
        self.add_dynamic_cons(dynamic_matrix_list, init_state, z)

    def add_lat_dynamic_cons(self, dynamic_matrix_list, init_state, theta_r):
        z = theta_r
        self.add_dynamic_cons(dynamic_matrix_list, init_state, z)

    def add_dynamic_cons(self, dynamic_matrix_list, init_state, theta_r):
        """
        Constraints for dynamic model
        x_(k+1) = A * x_k + B * u_k + D * z_k
        """
        for i in range(len(dynamic_matrix_list)):
            dynamic_matrix_dict = dynamic_matrix_list[i]
            if i == 0:
                # add additional constraints for initial state
                # x_0 - initial_state = 0
                self.add_matrix_eq_cons(
                    np.eye(dynamic_matrix_dict["A"].shape[0]),
                    np.zeros_like(dynamic_matrix_dict["B"]),
                    np.zeros_like(dynamic_matrix_dict["D"]),
                    self.x[:, i],
                    init_state,
                    0,
                    -1,
                    0,
                )
            self.add_matrix_eq_cons(
                dynamic_matrix_dict["A"],
                dynamic_matrix_dict["B"],
                dynamic_matrix_dict["D"],
                self.x[:, i + 1],
                self.x[:, i],
                self.u[i],
                i,
                theta_r[i],
            )

    def add_rule_cons(
        self,
        rule_constraints: Dict[
            BasePredicateEvaluator.predicate_name, PredicateConstraint
        ],
    ):
        """Add rule constraints"""
        for rule_con in rule_constraints.values():
            if rule_con.decision_variable:
                # rule constraints with binary variables
                # add binary variables
                self.create_binary_variable_in_cons(
                    rule_con.num_decision_variables,
                    rule_con.constraint_name,
                )
                self.add_binary_variables_cons(rule_con)
                self.add_binary_rule_constraint(rule_con, big_M=3000)
            else:
                # rule constraints without binary variables
                self.add_rule_constraint(rule_con)

    def add_matrix_eq_cons(self, A, B, D, x_left, x_right, u, t, z):
        """
        Add an equality constraint in form:
        x_left: x_(k+1)
        x_right: x_k
        u: u_k
        t: time step
        z: disturbance term (for lateral dynamic model)
        x = Ax+Bu+Dz
        """
        x_right = x_right.reshape([-1, 1])
        x_prop = A.dot(x_right) + B.dot(u) + D.dot(z)
        # Constraints for enforcing consistency of motion
        for i_x in range(x_right.size):
            self.model.addConstr(
                x_left[i_x] == x_prop[i_x, 0], "state_trans{}_at_time{}".format(i_x, t)
            )

    def create_binary_variable_in_cons(self, num_variables, constraint_name):
        """
        create binary variables based on rule constraints
        name of variable: delta_(constraint_name)_(index of variables)_(time step)
        """
        for i in range(num_variables):
            delta_tmp = list()
            for k in range(self.x_shape[1]):
                delta_tmp.append(
                    self.add_var(
                        typeofvar="binary",
                        name="delta_{}_{}_{}".format(constraint_name, i + 1, k),
                    )
                )
            self.delta["{}_{}".format(constraint_name, i + 1)] = np.array(delta_tmp)

    # TODO: add constraints for binary variables
    def add_binary_variables_cons(self, rule_constraint):
        for i in range(self.x_shape[1] - 1):
            pass

    def add_binary_rule_constraint(self, rule_constraint: PredicateConstraint, big_M):
        """
        add rule constraints using big-M method
        bounds for one of variables in state (longitudinal: x, v, a, j; lateral: d, theta, kappa, kappa_dot)
        constraint for predicates in_intersection_conflict_area and on_lanelet_with_type_intersection
        upper bound: x - big_M * delta - slack - ub <= 0
        lower bound: -x + big_M * delta - big_M + lb <= 0
        """
        # TODO: currently only consider predicates in_intersection_conflict_area and on_lanelet_with_type_intersection
        if rule_constraint.constraint_name in ["conflict_area", "intersection"]:
            for i in rule_constraint.time_step:
                time_step = rule_constraint.time_step.index(i)
                if rule_constraint.state_ub[time_step] != math.inf:
                    params_dict = {}
                    params_dict["vars"] = [
                        [1, self.x[rule_constraint.constraint_state, time_step]],
                        [
                            -big_M,
                            self.delta["{}_1".format(rule_constraint.constraint_name)][
                                time_step
                            ],
                        ],
                    ]
                    params_dict["constants"] = [-rule_constraint.state_ub[time_step]]
                    if self.slack is not None:
                        params_dict["vars"].append([-1, self.slack[1]])
                    self.add_ineq_cons(
                        params_dict,
                        "{}_ub_t{}".format(rule_constraint.constraint_name, time_step),
                    )
                if rule_constraint.state_lb[time_step] != -math.inf:
                    params_dict = {}
                    params_dict["vars"] = [
                        [-1, self.x[rule_constraint.constraint_state, time_step]],
                        [
                            big_M,
                            self.delta["{}_1".format(rule_constraint.constraint_name)][
                                time_step
                            ],
                        ],
                    ]
                    params_dict["constants"] = [
                        rule_constraint.state_lb[time_step],
                        -big_M,
                    ]
                    # TODO: currently do not consider slack variable for lower boundaries
                    #  because more unfixable situations occur when using slack variable for lower boundaries
                    if self.slack is not None:
                        pass
                    self.add_ineq_cons(
                        params_dict,
                        "{}_lb_t{}".format(rule_constraint.constraint_name, time_step),
                    )
        else:
            print("warning: no constraints added")

    def add_rule_constraint(self, rule_constraint: PredicateConstraint):
        """
        add rule constraints without binary variables
        bounds for one of variables in state (longtudinal: x, v, a, j; lateral: d, theta, kappa, kappa_dot)
        constraint for predicate in_intersection_conflict_area and on_lanelet_with_type_intersection
        upper bound: x - slack - ub <= 0
        lower bound: -x + lb <= 0
        """
        for i in rule_constraint.time_step:
            time_step = rule_constraint.time_step.index(i)
            if rule_constraint.state_ub[time_step] != math.inf:
                params_dict = {}
                params_dict["vars"] = [
                    [1, self.x[rule_constraint.constraint_state, time_step]],
                ]
                params_dict["constants"] = [-rule_constraint.state_ub[time_step]]
                if self.slack is not None:
                    params_dict["vars"].append([-1, self.slack[1]])
                self.add_ineq_cons(
                    params_dict,
                    "{}_ub_t{}".format(rule_constraint.constraint_name, time_step),
                )
            if rule_constraint.state_lb[time_step] != -math.inf:
                params_dict = {}
                params_dict["vars"] = [
                    [-1, self.x[rule_constraint.constraint_state, time_step]],
                ]
                params_dict["constants"] = [
                    rule_constraint.state_lb[time_step],
                ]
                # TODO: currently do not consider slack variable for lower boundaries
                if self.slack is not None:
                    pass
                self.add_ineq_cons(
                    params_dict,
                    "{}_lb_t{}".format(rule_constraint.constraint_name, time_step),
                )

    def add_collision_free_cons(
        self, collision_free_constraint: CollisionFreeConstraint
    ):
        """
        add collision free constraints
        upper bound: x - ub <= 0
        lower bound: -x + lb <= 0
        """
        for index in range(len(collision_free_constraint.time_step_lb)):
            params_dict = {}
            params_dict["vars"] = [
                [-1, self.x[0, collision_free_constraint.time_step_lb[index]]]
            ]
            params_dict["constants"] = [collision_free_constraint.lb[index]]
            self.add_ineq_cons(
                params_dict,
                "collision_free_cons_lb_at_time_{}".format(
                    collision_free_constraint.time_step_lb[index]
                ),
            )

        for index in range(len(collision_free_constraint.time_step_ub)):
            params_dict = {}
            params_dict["vars"] = [
                [1, self.x[0, collision_free_constraint.time_step_ub[index]]]
            ]
            params_dict["constants"] = [-collision_free_constraint.ub[index]]
            self.add_ineq_cons(
                params_dict,
                "collision_free_cons_ub_at_time_{}".format(
                    collision_free_constraint.time_step_ub[index]
                ),
            )

    def add_safe_distance_cons(
        self,
        safe_distance_modes: List[bool],
        pred_veh: Vehicle,
        velocity_samples,
        ti_cons: TIConstraint,
        tc,
    ):
        """
        constraint for safe distance
        safe_distance = safe_dis_0 + safe_dis_der_0 * (x[1] - v_sample)
        x[0] <= rear_s_pred_veh  - l/2 - wb/2 - safe_distance - 0.01
        """
        for k in range(len(safe_distance_modes) - 1):
            if safe_distance_modes[k]:
                param_dict = {}
                pred_veh_state = pred_veh.get_lon_state(k + tc)
                for i in range(len(velocity_samples)):
                    safe_dis_0 = calculate_safe_distance(
                        velocity_samples[i],
                        pred_veh_state.v,
                        -8,
                        -10.0,
                        0.3,
                    )
                    safe_dis_der_0 = derivative_safe_distance(
                        velocity_samples[i], -10, 0.3
                    )
                    param_dict["vars"] = [
                        [1, self.x[0, k + 1]],
                        [safe_dis_der_0, self.x[1, k + 1]],
                    ]
                    param_dict["constants"] = [
                        -(
                            pred_veh.rear_s(k + tc)
                            - ti_cons.length / 2
                            - ti_cons.wheelbase / 2
                            - safe_dis_0
                            + safe_dis_der_0 * velocity_samples[i]
                            - 1e-2
                        )
                    ]
                    self.add_ineq_cons(param_dict, "safe_dis_v_{}_t_{}".format(i, k))

    def add_lat_dis_cons(
        self,
        lat_dis_cons_matrix: List[Dict],
        x_ref_lat,
        d_min: np.ndarray,
        d_max: np.ndarray,
    ):
        """
        add constraints for lateral position
        """
        for i in range(self.x_shape[1] - 1):
            theta = x_ref_lat.reference[i].theta
            if list(d_min[i, :]) != [-np.inf, -np.inf, -np.inf]:
                S = lat_dis_cons_matrix[i]["S"]
                C = lat_dis_cons_matrix[i]["C"]
                E = lat_dis_cons_matrix[i]["E"]
                distance = S.dot(C.dot(self.x[:, i + 1]) + E.dot(theta))
                for k in range(distance.size):
                    params_dict = {}
                    params_dict["vars"] = [[-1, distance[k]]]
                    params_dict["constants"] = [d_min[i, k]]

                    # If slack variables are present, include them with negative coefficients
                    if self.slack is not None:
                        params_dict["vars"].append([-1, self.slack[1]])

                    self.add_ineq_cons(
                        params_dict, "lat_dist_cons_min_{}_{}".format(i + 1, k + 1)
                    )

            if list(d_max[i, :]) != [np.inf, np.inf, np.inf]:
                S = lat_dis_cons_matrix[i]["S"]
                C = lat_dis_cons_matrix[i]["C"]
                E = lat_dis_cons_matrix[i]["E"]
                distance = S.dot(C.dot(self.x[:, i + 1]) + E.dot(theta))

                for k in range(distance.size):
                    params_dict = {}
                    params_dict["vars"] = [[1, distance[k]]]
                    params_dict["constants"] = [-d_max[i, k]]

                    # If slack variables are present, include them with negative coefficients
                    if self.slack is not None:
                        params_dict["vars"].append([-1, self.slack[0]])

                    self.add_ineq_cons(
                        params_dict, "lat_dist_cons_max_{}_{}".format(i + 1, k + 1)
                    )

    def add_kappa_limit(self, kappa_lim):
        """add curvature constraint based on Kamm's circle"""
        for i in range(self.x_shape[1] - 1):
            params_dict = {}
            params_dict["vars"] = [[-1, self.x[2, i + 1]]]
            params_dict["constants"] = [-kappa_lim[i]]
            self.add_ineq_cons(params_dict, "lat_kappa_lim_min_{}".format(i + 1))

            params_dict = {}
            params_dict["vars"] = [[1, self.x[2, i + 1]]]
            params_dict["constants"] = [-kappa_lim[i]]
            self.add_ineq_cons(params_dict, "lat_kappa_lim_max_{}".format(i + 1))

    def add_ineq_cons(self, params_dict: dict, name):
        """
        add inequality constraint
        vars[1] * var[0] + constants <= 0
        vars[1]: variable
        vars[0]: multiplier
        """
        equation = LinExpr()
        for params in params_dict["vars"]:
            equation.add(params[1], params[0])
        if "constants" in params_dict.keys():
            for constant in params_dict["constants"]:
                equation.add(constant)
        self.model.addConstr(equation <= 0, name=name)

    def add_eq_cons(self, params_dict: dict, name):
        """
        add equality constraint
        vars[1] * var[0] + constants == 0
        vars[1]: multiplier
        vars[0]: variable
        """
        equation = LinExpr()
        for params in params_dict["vars"]:
            equation.add(params[1], params[0])
        if "constants" in params_dict.keys():
            for constant in params_dict["constants"]:
                equation.add(constant)
        self.model.addConstr(equation == 0, name=name)

    def costfunc_long(self, x_ref, weight):
        """cost function for the longitudinal planner"""
        long_costs = QuadExpr()
        weight_s = weight[0]
        weight_v = weight[1]
        weight_a = weight[2]
        weight_j = weight[3]
        weight_u = weight[4]
        weight_slack = weight[5]
        for i in range(self.x_shape[1]):
            diff_ref = LinExpr()
            diff_ref.add(self.x[0, i])
            diff_ref.addConstant(-x_ref.reference[i].s)
            long_costs.add(diff_ref * diff_ref, weight_s)

            diff_ref.clear()
            diff_ref.add(self.x[1, i])
            diff_ref.addConstant(-x_ref.reference[i].v)
            long_costs.add(diff_ref * diff_ref, weight_v)

            diff_ref.clear()
            diff_ref.add(self.x[2, i])
            diff_ref.addConstant(-x_ref.reference[i].a)
            long_costs.add(diff_ref * diff_ref, weight_a)

            diff_ref.clear()
            diff_ref.add(self.x[3, i])
            diff_ref.addConstant(-x_ref.reference[i].j)
            long_costs.add(diff_ref * diff_ref, weight_j)

        # fixme: additional punishment of the last state
        diff_ref = LinExpr()
        diff_ref.add(self.x[0, -1])
        long_costs.add(diff_ref * diff_ref, 100)

        for u in self.u:
            long_costs.add(u * u, weight_u)

        if self.slack is not None:
            for slack in self.slack:
                long_costs.add(slack, weight_slack)

        self.model.setObjective(long_costs, GRB.MINIMIZE)

    def costfunc_lat(self, x_ref, weight, d_reference, lat_cons: LateralConstraint):
        """cost function for the lateral planner"""

        cost_same_lane = False
        for proposition in lat_cons.select_proposition:
            for predicate in proposition.children:
                if predicate.base_name == PredInSameLane.predicate_name:
                    cost_same_lane = True

        lat_costs = QuadExpr()
        weight_d = weight[0]
        weight_theta = weight[1]
        weight_kappa = weight[2]
        weight_kappa_dot = weight[3]
        weight_u = weight[4]
        weight_robust = weight[5]
        weight_slack = weight[6]

        for i in range(1, self.x_shape[1]):
            diff_ref = LinExpr()
            diff_ref.add(self.x[0, i])
            diff_ref.addConstant(-2.3)
            lat_costs.add(diff_ref * diff_ref, weight_d)

            diff_ref.clear()
            diff_ref.add(self.x[1, i])
            diff_ref.addConstant(-round(x_ref.reference[i].theta, 2))
            lat_costs.add(diff_ref * diff_ref, weight_theta)

            diff_ref.clear()
            diff_ref.add(self.x[2, i])
            diff_ref.addConstant(-round(x_ref.reference[i].kappa, 2))
            lat_costs.add(diff_ref * diff_ref, weight_kappa)

            diff_ref.clear()
            diff_ref.add(self.x[3, i])
            diff_ref.addConstant(0)
            lat_costs.add(diff_ref * diff_ref, weight_kappa_dot)

            if cost_same_lane:
                diff_ref.clear()
                if (
                    lat_cons.d_min[i - 1, 0] != -np.inf
                    and lat_cons.d_max[i - 1, 0] != np.inf
                ):
                    d_rob = 0.5 * (lat_cons.d_min[i - 1, 0] + lat_cons.d_max[i - 1, 0])
                    diff_ref.add(self.x[0, i])
                    diff_ref.addConstant(-d_rob)
                    lat_costs.add(diff_ref * diff_ref, weight_robust)

        if self.slack is not None:
            for slack in self.slack:
                lat_costs.add(slack, weight_slack)

        for u in self.u:
            lat_costs.add(u * u, weight_u)

        self.model.setObjective(lat_costs, GRB.MINIMIZE)

    def solve(self):
        self.model.update()
        self.model.optimize()
        # todo: add a config value to determine whether to display model
        # self.model.display()

    def get_var_x(self):
        """
        Get state variables from solution
        """
        x_value = np.empty(self.x_shape)
        for i in range(self.x_shape[0]):
            for j in range(self.x_shape[1]):
                x_value[i, j] = self.get_var(self.x[i, j])
        return x_value

    def get_delta(self):
        """
        Get binary variables from solution
        """
        all_delta = list()
        for delta_name in self.delta:
            delta = self.delta[delta_name]
            delta_value = np.empty(delta.shape)
            for i in range(len(delta)):
                delta_value[i] = self.get_var(self.delta[delta_name][i])
            all_delta.append(delta_value)
        return all_delta

    def get_control_u(self):
        u_value = np.empty(self.u_shape)
        for i in range(self.u_shape[0]):
            u_value[i] = self.get_var(self.u[i])
        return u_value

    def get_slack_var(self):
        if self.slack is not None:
            slack_value = np.empty(self.slack_shape)
            for i in range(self.slack_shape[0]):
                slack_value[i] = self.get_var(self.slack[i])
            return slack_value
        else:
            return None

    def get_var(self, var):
        return var.x
