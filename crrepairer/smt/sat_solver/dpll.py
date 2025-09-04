import math
import functools

from sympy.logic.boolalg import is_cnf
from copy import deepcopy
from z3 import sat, unsat


class DPLL:
    def __init__(self, sympy_cnf: str, prop_nodes=None, tv_time_step=0):
        """
        Based on the pseudocode in Wikipedia page:
        https://en.wikipedia.org/wiki/DPLL_algorithm
        """
        assert is_cnf(sympy_cnf), (
            "<DPLL>: the given formula {} is not CNF or"
            " not in the sympy CNF standard".format(sympy_cnf)
        )
        self._cnf = self._assign_cnf(sympy_cnf)
        self._prop_nodes = prop_nodes
        self._tv_time_step = tv_time_step
        self._literals = self.get_literal(self._cnf, prop_nodes, tv_time_step)
        self._assign_true = set()
        self._assign_false = set()
        self._new_true = []
        self._new_false = []
        self._model = set()

    @property
    def model(self):
        return set.union(self._assign_true, self._assign_false)

    @property
    def literals(self):
        return self._literals

    @property
    def cnf(self):
        return self._cnf

    @staticmethod
    def get_literal(cnf, prop_nodes, tv_time_step: int):
        def robustness_degree(alp):
            rob_min_tv_h = 0
            node = next((x for x in prop_nodes if x.alphabet[-1] == alp[-1]), None)
            # TODO: FIXME: fix proposition sort. currently first consider propositions determined by the ego vehicle
            for predicate in node.children:
                if predicate.agent_placeholders == (1, 0):
                    rob_min_tv_h += 1
            rob_min_tv_h += abs(node.ttv_h_min)
            # print("<DPLL>: the robustness of instances in TV of alphabet {} is {}"
            #       .format(alp, prop_robust_all[prop_robust_all['alphabet'] == alp[-1]].robustness.values[tv_time_step]))
            return rob_min_tv_h

        literals = []
        for sub in cnf:
            split_cnf = sub.split()
            for lit in split_cnf:
                if lit[-1] not in literals and "~" + lit[-1] not in literals:
                    test = lit[-1]
                    literals.append(lit)
        # use robustness as heuristics to rank the literals
        if prop_nodes is not None and tv_time_step is not math.inf:
            return sorted(literals, key=robustness_degree)
        else:
            return literals

    @staticmethod
    def _assign_cnf(sympy_cnf):
        return (
            sympy_cnf.replace("(", "")
            .replace("~~", "")
            .replace(")", "")
            .replace("|", "")
            .split(" & ")
        )

    def update_cnf(self, cnf):
        self._cnf = self._assign_cnf(cnf)
        self._literals = self.get_literal(
            self._cnf, self._prop_nodes, self._tv_time_step
        )
        self._assign_true = set()
        self._assign_false = set()
        self._new_true = []
        self._new_false = []

    def solve(self):
        return self._solve(deepcopy(self._cnf))

    def back_tracking(self):
        for i in self._new_true:
            self._assign_true.remove(i)
        for i in self._new_false:
            self._assign_false.remove(i)

    def _solve(self, cnf):
        units = [i for i in cnf if len(i) < 3]
        units = list(set(units))
        self._new_true = []
        self._new_false = []
        self._assign_true = set(self._assign_true)
        self._assign_false = set(self._assign_false)
        if len(units):
            cnf = [clause.replace("~~", "") for clause in cnf]
            cnf = self.unit_propagation(cnf, units)
        if len(cnf) == 0:
            # if \phi is a consistent set of literals
            return sat
        if sum(len(clause) == 0 for clause in cnf):
            # if \phi contains an empty clause
            self.back_tracking()
            return unsat
        literals = self.get_literal(cnf, self._prop_nodes, self._tv_time_step)
        lit = self.choose_literal(literals)
        # print('<DPLL>: literal ({}) is selected'.format(lit))
        if self._solve(deepcopy(cnf) + [lit]) == sat:
            return sat
        elif self._solve(deepcopy(cnf) + ["~" + lit]) == sat:
            return sat
        else:
            self._assign_true = set()
            self._assign_false = set()
            return unsat

    def choose_literal(self, literals):
        return literals[0]

    def unit_propagation(self, cnf, units):
        for unit in units:
            if "~" in unit:
                self._assign_false.add(unit)
                self._new_false.append(unit)
                i = 0
                while True:
                    if unit in cnf[i]:
                        cnf.remove(cnf[i])
                        i -= 1
                    elif unit[-1] in cnf[i]:
                        cnf[i] = cnf[i].replace(unit[-1], "").strip()
                        if "  " in cnf[i]:
                            cnf[i] = cnf[i].replace("  ", " ")
                    i += 1
                    if i >= len(cnf):
                        break
            else:
                self._assign_true.add(unit)
                self._new_true.append(unit)
                i = 0
                while True:
                    if "~" + unit in cnf[i]:
                        cnf[i] = cnf[i].replace("~" + unit, "").strip()
                        if "  " in cnf[i]:
                            cnf[i] = cnf[i].replace("  ", " ")
                    elif unit in cnf[i]:
                        cnf.remove(cnf[i])
                        i -= 1
                    i += 1
                    if i >= len(cnf):
                        break
        return cnf


if __name__ == "__main__":
    dpll_solver = DPLL("a & ~a")
    print(dpll_solver.solve())
    print(dpll_solver.model)
