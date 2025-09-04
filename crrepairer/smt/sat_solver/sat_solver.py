from crrepairer.smt.sat_solver.dpll import DPLL
from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.smt import construct_cnf


class SATSolver:
    def __init__(self, rule_monitor: STLRuleMonitor, config: RepairerConfiguration):
        # nnf is constructed with the monitor
        # self._formula = construct_nnf(rule_monitor.sat_formula)
        self._formula = construct_cnf(rule_monitor.sat_formula)
        print("* \t<SATSolver>: the formula in CNF is {}".format(self._formula))
        self._prop_nodes = rule_monitor.proposition_nodes
        self._prop_robust_all = rule_monitor.rob_abstraction
        self._init_assign = list()
        self._dpll_solver = DPLL(
            self._formula, self._prop_nodes, rule_monitor.tv_time_step
        )
        self._dpll_model = None

        self._config = config

    @property
    def formula(self):
        return self._formula

    @property
    def initial_assignment(self):
        return self._init_assign

    def solve(self):
        """
        SAT Solver.
        There are multiple choices for the SAT solver. *Pysat* supports the DIMACS CNF as inputs, *z3*: a theorem solver
        from Microsoft Research. Here we use *sympy* for its easy-to-use interface
        """
        self._dpll_solver.update_cnf(self._formula)
        sat_result = self._dpll_solver.solve()
        return sat_result

    def model(self) -> (list, str):
        """
        return a satisfiable proposition - based on robustness
        """
        self._dpll_model = self._dpll_solver.model
        prop_list = list()
        for m in list(self._dpll_model):
            sel_prop_node = next(
                (
                    prop_node
                    for prop_node in self._prop_nodes
                    if prop_node.alphabet[-1] == m[-1]
                ),
                None,
            )
            if sel_prop_node:
                sel_prop_node.alphabet = m  # Assign m to the alphabet attribute
            prop_list.append(sel_prop_node)
        print("* \t<SATSolver>: model is {}".format(self._dpll_model))
        return prop_list, self._dpll_model

    def update_formula(self):
        """
        Based on the syntax for sympy, the SAT formula is updated by negating the unsatisfiable abstraction:
        phi_SAT = phi_SAT and (not abs)
        """
        if self._formula[0] is not "(":
            self._formula = "(" + self.formula + ")"
        # generate counterexample
        counter_ex = "~" + list(self._dpll_model)[0]
        if len(list(self._dpll_model)) > 1:
            counter_ex = "(" + counter_ex
            for atom in list(self._dpll_model)[1:]:
                counter_ex += " | ~" + atom
            counter_ex += ")"
        self._formula += " & " + counter_ex
        print("* \t<SATSolver>: the formula is updated to {}".format(self._formula))
