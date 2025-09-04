import sympy as sp
import re


class NNFFormula:
    def __init__(self, kind, left=None, right=None, prop=None):
        self.kind = kind  # Can be 'prop', 'neg', 'and', 'or'
        self.left = left  # Left operand for binary operators
        self.right = right  # Right operand for binary operators
        self.prop = prop  # Atomic proposition (for 'prop' type)

    def __repr__(self):
        if self.kind == "prop":
            return self.prop
        elif self.kind == "neg":
            return f"¬{self.left}"
        elif self.kind == "and":
            return f"({self.left} ∧ {self.right})"
        elif self.kind == "or":
            return f"({self.left} ∨ {self.right})"

    def compute_tv(self, tv_values: dict):
        """Recursively compute the Time-to-Violation (TV) based on the given values."""
        if self.kind == "prop":
            # Look up the TV value for the atomic proposition
            return tv_values[self.prop]
        elif self.kind == "neg":
            # For negation, we now explicitly expect the negated form in the tv_values
            neg_prop = (
                f"~{self.left.prop}"  # Representing negation as ~prop in tv_values
            )
            if neg_prop not in tv_values:
                raise ValueError(
                    f"TV for negation of {self.left.prop} (as ~{self.left.prop}) not provided in tv_values"
                )
            return tv_values[neg_prop]
        elif self.kind == "and":
            tv_left = self.left.compute_tv(tv_values)
            tv_right = self.right.compute_tv(tv_values)
            return min(tv_left, tv_right)
        elif self.kind == "or":
            tv_left = self.left.compute_tv(tv_values)
            tv_right = self.right.compute_tv(tv_values)
            return max(tv_left, tv_right)

    def compute_tv_list(self, tv_values: dict):
        """Recursively compute the Time-to-Violation (TV) series based on the given time series."""

        # Handle atomic propositions
        if self.kind == "prop":
            # Get the time series for the proposition
            return tv_values[self.prop]

        # Handle negation
        elif self.kind == "neg":
            neg_prop = (
                f"~{self.left.prop}"  # Representing negation as ~prop in tv_values
            )
            if neg_prop not in tv_values:
                raise ValueError(
                    f"TV for negation of {self.left.prop} (as ~{self.left.prop}) not provided in tv_values"
                )
            return tv_values[neg_prop]

        # Handle conjunction (element-wise min between left and right time series)
        elif self.kind == "and":
            tv_left = self.left.compute_tv(tv_values)
            tv_right = self.right.compute_tv(tv_values)
            return [min(l, r) for l, r in zip(tv_left, tv_right)]

        # Handle disjunction (element-wise max between left and right time series)
        elif self.kind == "or":
            tv_left = self.left.compute_tv(tv_values)
            tv_right = self.right.compute_tv(tv_values)
            return [max(l, r) for l, r in zip(tv_left, tv_right)]


def parse_nnf_formula(input_str):
    """Parse the NNF formula string and return a structured Formula object."""
    # Remove spaces from the input string
    input_str = input_str.replace(" ", "")

    # Handle atomic propositions
    if re.match(r"^~?[a-z]$", input_str):
        if input_str.startswith("~"):
            return NNFFormula("neg", left=NNFFormula("prop", prop=input_str[1:]))
        else:
            return NNFFormula("prop", prop=input_str)

    # Find the main operator in the formula
    depth = 0
    for i, char in enumerate(input_str):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and char in {"&", "|"}:
            left = parse_nnf_formula(input_str[:i])
            right = parse_nnf_formula(input_str[i + 1 :])
            if char == "&":
                return NNFFormula("and", left=left, right=right)
            elif char == "|":
                return NNFFormula("or", left=left, right=right)

    # Remove surrounding parentheses
    if input_str.startswith("(") and input_str.endswith(")"):
        return parse_nnf_formula(input_str[1:-1])

    raise ValueError(f"Invalid formula format: {input_str}")


def stl2sympy(input_formula: str):
    return (
        input_formula.replace("and", "&")
        .replace("or", "|")
        .replace("!", "~")
        .replace("implies", ">>")
    )


def construct_cnf(stl_formula):
    """
    Construct Conjunctive Normal Form (CNF) using sympy - first needs to convert the formula to sp's interface.
    """
    if isinstance(stl_formula, str):
        sp_formula = stl2sympy(stl_formula)
    else:
        sp_formula = stl_formula
    cnf_formula = str(sp.to_cnf(sp_formula))
    return cnf_formula


def construct_dnf(stl_formula):
    """
    Construct Disjunctive Normal Form (DNF) using sympy - first needs to convert the formula to sp's interface.
    """
    if isinstance(stl_formula, str):
        sp_formula = stl2sympy(stl_formula)
    else:
        sp_formula = stl_formula
    dnf_formula = str(sp.to_dnf(sp_formula))
    return dnf_formula


def construct_nnf(stl_formula):
    """
    Construct Negation Normal Form (NNF) using sympy - first needs to convert the formula to sp's interface.
    """
    sp_formula = stl2sympy(stl_formula)
    nnf_formula = sp.to_nnf(sp.simplify(sp_formula))
    return nnf_formula
