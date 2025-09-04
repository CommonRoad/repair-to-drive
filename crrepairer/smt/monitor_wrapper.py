import functools
import math
import string
from itertools import product
from typing import Iterable, Union, Tuple, Any, List, Dict, Optional
from collections import defaultdict
import numpy as np
import dataclasses
from dataclasses import dataclass, field
import copy
from difflib import SequenceMatcher
from multiprocessing import Process, Queue

import re
from crmonitor.evaluation.proposition_evaluation import PropositionRuleEvaluator
from crmonitor.evaluation.evaluation import (
    get_evaluation_config,
    create_ego_vehicle_param,
)
from crmonitor.common.world import World, get_world_config
from crmonitor.common.config import get_traffic_rule_config
from crmonitor.rule.rule_node import PredicateNode

from crrepairer.utils.configuration import (
    RepairerConfiguration,
    ScenarioType,
    MonitorType,
)
from crrepairer.utils.smt import construct_nnf, parse_nnf_formula, NNFFormula

from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as Cfg


@dataclass
class PropositionNode:
    name: str
    alphabet: str
    source_rule: str
    children: List["PredicateNode"] = field(default_factory=list)
    ttv_value: Optional[float] = None
    ttv_h_min: Optional[float] = None  # Optional, initialized as None

    def set_ttv_values(self, ttv_value: float, ttv_h_min: float):
        """Method to set the ttv_value and ttv_h_min later."""
        self.ttv_value = ttv_value
        self.ttv_h_min = ttv_h_min


class STLRuleMonitor:
    def __init__(
        self,
        config: RepairerConfiguration,
    ):
        # update the world configuration for repairing purposes
        world_config = get_world_config()
        traffic_rules_config = get_traffic_rule_config()

        world_config["scenario"] = self.scenario_type = traffic_rules_config[
            "traffic_rules_param"
        ]["mpr_scenario"] = config.repair.scenario_type
        if self.scenario_type == ScenarioType.INTERSECTION:
            world_config["intersection_road_network_param"][
                "map_type"
            ] = config.repair.intersection_type
        traffic_rules_config["traffic_rules_param"]["use_mpr"] = config.repair.use_mpr

        self._world: World = World.create_from_scenario(
            config.scenario, config=world_config
        )
        self._vehicle_id = config.repair.ego_id
        self.multiproc = config.repair.multiproc
        self.mpr = config.repair.use_mpr
        self._rules = config.repair.rules
        self._rule_eval = []
        self._start_time_step = self._world.vehicle_by_id(self._vehicle_id).start_time
        self._world.vehicle_by_id(self._vehicle_id).vehicle_param = (
            create_ego_vehicle_param(
                get_evaluation_config().get("ego_vehicle_param"), self._world.dt
            )
        )
        for rule in self._rules:
            prop_rule_eval = PropositionRuleEvaluator.create_from_config(
                self._world,
                self._vehicle_id,
                rule,
                traffic_rules_config=traffic_rules_config,
            )
            self._rule_eval.append(prop_rule_eval)
        if len(self._rule_eval) == 1:
            self.multiproc = False
        (
            self.rob_rule,
            self.rob_predicate,
            self.rob_abstraction,
            self.abstraction_names,
            self.other_ids,
            self.all_props_all_ids_all,
            self.all_rules_all_ids_all,
        ) = self.evaluate_initially()
        # initialize the propositional nodes
        self._prop_nodes = self._initialize_prop_nodes()
        # generate the SAT formula in the NNF

        self.sat_formula, self.sat_formula_sep = self.obtain_sat_formula_in_nnf()
        print("===== formula in NNF: ", self.sat_formula)
        # obtain the time-to-violation using the way written in the Journal paper
        (
            self._violated_rules,
            self.min_rule_idx,
            self._tv,
            self.rule_to_tv,
            self.rule_to_other_id,
        ) = self._cal_tv_def()

        self._future_time_step = self.search_future_time_step()[self.min_rule_idx]
        self._update_prop_nodes()

        # # obtain the time-to-violation
        # (
        #     self._violated_rules,
        #     self.min_rule_idx,
        #     self._tv,
        #     self.rule_to_tv,
        #     self.rule_to_other_id,
        # ) = self._cal_tv_initial()

        # todo: multiple targets
        self._other_id = [
            other_id
            for other_id in self.rule_to_other_id.values()
            if other_id != config.repair.ego_id
        ]

        print("# =========== Traffic Rule Monitor ========== #")
        for rule in self._violated_rules:
            print(
                "\tThe ego vehicle (ID: {})'s initial trajectory violates the traffic rule: {}.".format(
                    self._vehicle_id, rule
                )
            )
            print(
                "\tViolation occurred at time step: {}, with respect to vehicle ID: {}.".format(
                    self.rule_to_tv[rule], self.rule_to_other_id[rule]
                )
            )
        rob_flat = self.rob_rule.flatten()
        lines = [
            ", ".join(
                f"step {i + 1}: {val:.5f}"
                for i, val in enumerate(rob_flat[start : start + 5], start=start)
            )
            for start in range(0, len(rob_flat), 5)
        ]

        print("============ Robustness values: ============")
        for line in lines:
            print(line)
        print("# =========================================== #")

    @property
    def tv_time_step(self) -> Union[int, float]:
        return self._tv  # - self._future_time_step

    @property
    def tv_time_step_with_future(self) -> Union[int, float]:
        return self._tv

    @property
    def future_time_step(self) -> Union[int, float]:
        return self._future_time_step

    @property
    def start_time_step(self):
        return self._start_time_step

    @property
    def other_id(self) -> int:
        if len(self._other_id) == 0:
            return self._vehicle_id
        else:
            return self._other_id[0]

    @property
    def vehicle_id(self) -> int:
        return self._vehicle_id

    @property
    def world(self) -> World:
        return self._world

    @property
    def type(self):
        return MonitorType.STL

    @property
    def rule_eval(self):
        return self._rule_eval

    @property
    def proposition_nodes(self) -> List[PropositionNode]:
        return self._prop_nodes

    def obtain_sat_formula_in_nnf(self):
        """
        For all propositions, find the overlapping subsequences in the rule string
        and replace them with the alphabet symbols corresponding to the proposition nodes.
        """
        subformula_dict = {}
        subformula_list = []
        prev_idx = 0

        for i, evaluator in enumerate(self._rule_eval):
            rule_node = evaluator._rule
            sat_formula = (
                rule_node.children[0].rule_str
                if len(rule_node.children) == 1
                else rule_node.rule_str
            )

            # Normalize formula strings: Replace logical symbols and keywords
            # 'eventually' is replaced by 'once' because of the same replacement in rtamt
            sat_formula = sat_formula.replace("not", "!").replace("eventually", "once")

            # Find the relevant proposition nodes based on the robustness abstraction
            clear_rob_abs = self.rob_abstraction[i][
                self.rob_abstraction[i] == self.rob_abstraction[i]
            ]
            length = int(clear_rob_abs.shape[0] / self.rob_abstraction[i].shape[0])
            props_of_rule = self._prop_nodes[prev_idx : prev_idx + length]
            prev_idx += length

            for prop_node in props_of_rule:
                prop_node_name = self._normalize_prop_name(prop_node.name)

                # Match and replace the proposition name in the formula
                matches = SequenceMatcher(
                    None, sat_formula, prop_node_name, autojunk=True
                ).get_matching_blocks()
                clean_matches = [match for match in matches if match.size > 1]
                if clean_matches:
                    first_index = clean_matches[0].a
                    last_index = clean_matches[-1].a + clean_matches[-1].size
                    to_replace = re.escape(sat_formula[first_index:last_index])

                    # Avoid replacing incorrect propositions
                    pattern = rf"(?<!\]\(){to_replace}"
                    sat_formula = re.sub(pattern, prop_node.alphabet, sat_formula)

            subformula_list.append(f"({sat_formula})")
            subformula_dict[rule_node.name] = construct_nnf(sat_formula)

        # Join the subformulas with "and" operators
        sat_formula = " and ".join(subformula_list)

        # Return the formula in NNF
        return construct_nnf(sat_formula), subformula_dict

    def _normalize_prop_name(self, prop_name):
        """
        Normalize proposition names that start with 'once' or contain temporal predicates.
        Ensure that if a name starts and ends with parentheses, only the outermost pair is removed.
        """
        # Handle propositions that are predicates (starting with "once" and similar patterns)
        if prop_name.startswith("once") and prop_name[5:6] == prop_name[7:8]:
            prop_name = prop_name.replace(prop_name[0:9], "")

        # Remove "once[...]" pattern from the proposition name
        pattern = r"once\[(.*?)\]"
        matches = re.findall(pattern, prop_name)
        if len(matches) == 2:
            to_delete = f"once[{matches[1]}]("
            prop_name = prop_name.replace(to_delete, "")
            # Remove only the last closing parenthesis if it exists
            if prop_name.endswith(")"):
                prop_name = prop_name[:-1]

        # Remove one pair of parentheses from both ends if both exist
        while prop_name.startswith("(") and prop_name.endswith(")"):
            prop_name = prop_name[1:-1]  # Remove only the outermost pair of parentheses

        return prop_name

    @property
    @functools.lru_cache(128)
    def prop_robust_all(self):
        return self.rob_abstraction  # [self._violated_rule_idx]

    @property
    @functools.lru_cache(128)
    def prop_robust_ttv(self):
        return self.rob_abstraction[self.min_rule_idx][self._tv - self._start_time_step]

    @staticmethod
    def infinite_alphabet():
        """Generate an infinite sequence of alphabetic labels like a, b, ..., z, aa, ab, ..., zz, aaa, ..."""

        for size in range(1, 100):  # Choose a high enough range for your needs
            for letters in product(string.ascii_lowercase, repeat=size):
                yield "".join(letters)

    def _initialize_prop_nodes(self):
        """
        Construct 'nodes' for propositions for better backward compatibility.

        Returns:
        prop_nodes (List[PropositionNode]): List of proposition nodes
        """
        all_prop_names = self.abstraction_names[:, 0]

        prop_nodes = []
        alphabet_gen = (
            self.infinite_alphabet()
        )  # Initialize the infinite alphabet generator

        # Get the indices of non-empty proposition names
        non_empty_indices = np.transpose(all_prop_names.nonzero())

        # Loop through each index of `all_prop_names` to create proposition nodes
        for idx in non_empty_indices:
            prop_name = all_prop_names[tuple(idx)]  # Access the valid proposition name

            # Get the next available alphabet character/sequence
            prop_alphabet = next(alphabet_gen)

            # Construct the PropositionNode without ttv_value and ttv_h_min, since they are optional
            proposition = PropositionNode(
                name=prop_name,
                alphabet=prop_alphabet,
                source_rule=self._rule_eval[idx[0]]._rule.name,
            )

            # Append the constructed node to the list
            prop_nodes.append(proposition)

        return prop_nodes

    def _update_prop_nodes(self):
        """
        Update the ttv_value and ttv_h_min of the proposition nodes.
        """

        def retrieve_preds(node, liste):
            # Method for retrieving PredicateNodes, which are to be used in determining maneuvers
            for child in node.children:
                if hasattr(child, "latest_value"):
                    liste.append(child)
                else:
                    retrieve_preds(child, liste)

        all_prop_names = self.abstraction_names[:, 0]
        all_prop_robs = np.full(
            all_prop_names.shape, np.nan
        )  # Initialize with NaN for safety

        tv_prop_robs = np.full(all_prop_names.shape, np.nan)
        all_pre_rob_grad = np.empty(all_prop_names.shape[0], dtype=object)

        # Initialize prop_index for prop_node assignments
        prop_index = 0

        for i in range(all_prop_names.shape[0]):  # Iterate over rows
            pred_nodes = []
            retrieve_preds(self._rule_eval[i]._rule, pred_nodes)

            # all_pre_rob_grad should have the same length as all_prop_names.shape[0]
            rob_index = self.rule_to_tv[self._rules[i]] - self._start_time_step
            if 0 <= rob_index < len(self.rob_predicate[i]):
                all_pre_rob_grad[i] = self.rob_predicate[i][rob_index]
                # print out the gradient of the predicate
            else:
                all_pre_rob_grad[i] = (
                    np.nan
                )  # Assign NaN if the index is out of range or invalid

            print("=====================================")
            for j in range(all_prop_names.shape[1]):  # Iterate over columns

                if prop_index < len(
                    self._prop_nodes
                ):  # Ensure prop_index does not exceed the number of prop_nodes
                    # tv is now self.rule_to_tv[self._rules[i]]
                    # Get the property name and sequence
                    prop_name = all_prop_names[i, j]
                    other_id = self.rule_to_other_id[self._rules[i]]
                    seq = self.all_props_all_ids_all[i][prop_name][other_id]

                    index = str(self.sat_formula).find(
                        self._prop_nodes[prop_index].alphabet
                    )
                    sign = (
                        "~"
                        if index > 0 and str(self.sat_formula)[index - 1] == "~"
                        else None
                    )
                    # Check if the sequence is empty or if the slicing would go out of bounds
                    if (
                        prop_name.startswith("once")
                        and prop_name[5:6] == prop_name[7:8]
                    ):
                        future_index = self._future_time_step
                    else:
                        future_index = 0
                    if seq and (self._tv - self._start_time_step) < len(seq):
                        # Safely slice the sequence from the desired index to the end
                        # Calculate the relevant subsequence

                        subseq = seq[self._tv - self._start_time_step + future_index :]

                        # Assign to all_prop_robs based on the presence of a negation sign
                        # G(a) and G(not a) are handled differently
                        # Assign to all_prop_robs based on the presence of a negation sign
                        if sign:
                            all_prop_robs[i, j] = min(-val for val in subseq)
                        else:
                            all_prop_robs[i, j] = min(subseq)
                    else:
                        # If sequence is empty or the slicing index is out of bounds, set to -1
                        all_prop_robs[i, j] = -1

                    # Calculate the index for tv_prop_robs safely
                    tv_index = self.rule_to_tv[self._rules[i]] - self._start_time_step
                    if (
                        0 <= tv_index + future_index < len(seq)
                    ):  # Ensure tv_index is within valid range
                        tv_prop_robs[i, j] = seq[tv_index + future_index]
                    else:
                        tv_prop_robs[i, j] = (
                            -1
                        )  # Assign -1 if the index is out of range or invalid

                    # Update the property node with ttv_value and ttv_h_min
                    self._prop_nodes[prop_index].set_ttv_values(
                        ttv_value=tv_prop_robs[i, j], ttv_h_min=all_prop_robs[i, j]
                    )
                    print(
                        f"Proposition '{prop_name}' ({self._prop_nodes[prop_index].alphabet})"
                        f" of rule {self._prop_nodes[prop_index].source_rule}"
                        f" has ttv_value: {tv_prop_robs[i, j]} and ttv_h_min: {all_prop_robs[i, j]}"
                    )
                    for pred in pred_nodes:
                        if "g0" not in all_prop_names[tuple([i, j])]:
                            if pred.name in all_prop_names[tuple([i, j])]:
                                # Add the missing values (latest_value, mpr_gradient)
                                if all_pre_rob_grad[i] is not np.nan:
                                    pred.latest_value, pred.mpr_gradient = (
                                        all_pre_rob_grad[i][pred.name]
                                    )
                                else:
                                    pred.latest_value = None
                                    pred.mpr_gradient = None
                                self._prop_nodes[prop_index].children.append(pred)
                        else:
                            # Handle case when "g0" is present in the prop_name
                            other_props = np.delete(all_prop_names[i], j, 0)
                            if not any(
                                [
                                    pred.name in p_name
                                    for p_name in other_props
                                    if p_name
                                ]
                            ):
                                if all_pre_rob_grad[i] is not np.nan:
                                    pred.latest_value, pred.mpr_gradient = (
                                        all_pre_rob_grad[i][pred.name]
                                    )
                                else:
                                    pred.latest_value = None
                                    pred.mpr_gradient = None
                                self._prop_nodes[prop_index].children.append(pred)
                else:
                    raise IndexError(
                        f"prop_index {prop_index} exceeds the number of propositional nodes."
                    )
                prop_index += 1  # Increment the prop_index for the next property node
            print("=====================================")

    def search_future_time_step(self):
        future_time_step = np.zeros(len(self.rule_eval), dtype=int)
        for i in range(len(future_time_step)):
            prop_robust_rule = self.prop_robust_all[i, :, :]
            for j in range(prop_robust_rule.shape[0]):
                if np.any(np.isinf(prop_robust_rule[j, :])):
                    future_time_step[i] += 1
                else:
                    break
        return future_time_step

    def evaluate_initially(self):
        """
        Evaluate whether the ego vehicle disobeys traffic rules

        Update: Now uses the get_propositions method of the STLRuleEvaluator to obtain the rule, predicate, and
        proposition robustness values. The values are only obtained for the highest non-conforming vehicle at
        each time step, the id of which is kept for better backward compatibility and higher verbosity.

        Returns:
        df_rule (np.ndarray): DF constructed of the rule robustness at each timestep
        df_pred (np.ndarray): DF constructed of each predicate robustness at each timestep for given other_id
        df_prop (np.ndarray): DF constructed of each proposition robustness at each timestep for given other_id
        other_ids List(Tuple): Vehicle ids w.r.t which the rule robustness was calculated
        """
        rule_rob_all = []
        prop_rob_all = []
        prop_names_all = []
        pred_rob_all = []
        other_ids_all = []

        all_props_all_ids_all = []
        all_rules_all_ids_all = []
        if self.multiproc and not self.mpr:
            rule_ids = []
            queue = Queue()
            processes = [
                Process(target=self.multiproc_evaluate, args=[i, queue])
                for i in range(len(self._rule_eval))
            ]
            for p in processes:
                p.start()
            for _ in range(len(self._rule_eval)):
                res = queue.get()
                rule_rob_all.append(res["rule"])
                prop_rob_all.append(res["prop"])
                prop_names_all.append(res["prop_name"])
                pred_rob_all.append(res["pred"])
                other_ids_all.append(res["other"])
                rule_ids.append(res["index"])
                all_props_all_ids_all.append(res["all_props"])
                all_rules_all_ids_all.append(res["all_rules"])
            for p in processes:
                p.join()
            self._rule_eval = [self._rule_eval[i] for i in rule_ids]
            self._rules = [self._rules[i] for i in rule_ids]

        else:
            for evaluator in self._rule_eval:
                rule_rob = []
                prop_rob = []
                prop_names = []
                pred_rob = []
                other_ids = []

                all_rules_all_ids = {}
                all_props_all_ids = {}
                for _ in range(
                    evaluator.ego_vehicle.start_time, evaluator.ego_vehicle.end_time + 1
                ):
                    rule_rob.append(evaluator.update())
                    other_ids.append(evaluator.other_ids)
                    prop, other_id_props, all_rules_all_pre, _ = (
                        evaluator.get_propositions_all()
                    )
                    # Combine processing of prop and all_values_all_ids
                    for prop_name, vehicle_dict in prop.items():
                        if prop_name not in all_props_all_ids:
                            all_props_all_ids[prop_name] = {}

                        for vid, rob in vehicle_dict.items():
                            # Ensure that all_props_all_ids[prop_name][vid] is a list
                            if vid not in all_props_all_ids[prop_name]:
                                all_props_all_ids[prop_name][vid] = []
                            # Populate all_props_all_ids
                            all_props_all_ids[prop_name][vid].append(rob)

                    for vid in all_rules_all_pre.keys():
                        # Populate all_rules_all_ids
                        if vid not in all_rules_all_ids:
                            all_rules_all_ids[vid] = []
                        all_rules_all_ids[vid].append(all_rules_all_pre[vid])
                    if other_id_props:
                        prop_names.append(
                            [prop_name for prop_name in other_id_props.keys()]
                        )
                        prop_rob.append(
                            [
                                other_id_props[prop_name]
                                for prop_name in other_id_props.keys()
                            ]
                        )
                    else:
                        try:
                            prop_names.append(prop_names[0])
                            prop_rob.append([-np.inf] * len(prop_rob[0]))
                        except:
                            prop_names.append([])
                            prop_rob.append([])
                    pred = evaluator.get_predicates()
                    if pred:
                        mpr_grad = evaluator.get_mpr_gradient()
                        pred_rob.append(
                            {key: [pred[key], mpr_grad[key]] for key in pred}
                        )
                    else:
                        pred_rob.append([])

                rule_rob_all.append(np.array(rule_rob, dtype=np.float64))
                prop_rob_all.append(np.array(prop_rob, dtype=np.float64))
                prop_names_all.append(np.array(prop_names, dtype=object))
                pred_rob_all.append(pred_rob)
                other_ids_all.append(other_ids)
                all_props_all_ids_all.append(all_props_all_ids)
                all_rules_all_ids_all.append(all_rules_all_ids)

        assert len(rule_rob_all) == len(self._rule_eval)
        max_n_props = max([p.shape[1] for p in prop_rob_all])
        for idx, prop_array in enumerate(prop_rob_all):
            if prop_array.shape[1] < max_n_props:
                prop_rob_all[idx] = np.pad(
                    prop_array,
                    ((0, 0), (0, max_n_props - prop_array.shape[1])),
                    "constant",
                    constant_values=np.nan,
                )
                prop_names_all[idx] = np.pad(
                    prop_names_all[idx],
                    ((0, 0), (0, max_n_props - prop_array.shape[1])),
                    "constant",
                    constant_values=np.nan,
                )
        return (
            np.array(rule_rob_all),
            np.array(pred_rob_all),
            np.array(prop_rob_all),
            np.array(prop_names_all),
            other_ids_all,
            all_props_all_ids_all,
            all_rules_all_ids_all,
        )

    def multiproc_evaluate(self, index, q):
        evaluator = self._rule_eval[index]
        rule_rob = []
        prop_rob = []
        prop_names = []
        pred_rob = []
        other_ids = []
        all_rules_all_ids = {}
        all_props_all_ids = {}

        for _ in range(
            evaluator.ego_vehicle.start_time, evaluator.ego_vehicle.end_time + 1
        ):
            rule_rob.append(evaluator.update())
            other_ids.append(evaluator.other_ids)

            prop, other_id_props, all_rules_all_pre, _ = (
                evaluator.get_propositions_all()
            )
            # Combine processing of prop and all_values_all_ids
            for prop_name, vehicle_dict in prop.items():
                if prop_name not in all_props_all_ids:
                    all_props_all_ids[prop_name] = {}

                for vid, rob in vehicle_dict.items():
                    # Ensure that all_props_all_ids[prop_name][vid] is a list
                    if vid not in all_props_all_ids[prop_name]:
                        all_props_all_ids[prop_name][vid] = []
                    # Populate all_props_all_ids
                    all_props_all_ids[prop_name][vid].append(rob)

                for vid in all_rules_all_pre.keys():
                    # Populate all_rules_all_ids
                    if vid not in all_rules_all_ids:
                        all_rules_all_ids[vid] = []
                    all_rules_all_ids[vid].append(all_rules_all_pre[vid])
            if other_id_props:
                prop_names.append([prop_name for prop_name in other_id_props.keys()])
                prop_rob.append(
                    [other_id_props[prop_name] for prop_name in other_id_props.keys()]
                )
            else:
                prop_names.append([])
                prop_rob.append([])
            pred = evaluator.get_predicates()
            if pred:
                mpr_grad = evaluator.get_mpr_gradient()
                pred_rob.append({key: [pred[key], mpr_grad[key]] for key in pred})
            else:
                pred_rob.append({})
        return_dict = {
            "rule": np.array(rule_rob, dtype=np.float64),
            "other": other_ids,
            "prop": np.array(prop_rob, dtype=np.float64),
            "prop_name": np.array(prop_names, dtype=object),
            "pred": np.array(pred_rob, dtype=object),
            "index": index,
            "all_props": all_props_all_ids,
            "all_rules": all_rules_all_ids,
        }
        q.put(return_dict)

    def evaluate_consecutively(self, world, reset_time, bool_mode=True):
        """
        Evaluate the updated vehicle states (boolean assignments) in order to speed up the evaluation progress
        """
        world_state = copy.copy(world)
        rule_rob_all = []
        other_ids_all = []
        for evaluator in self._rule_eval:
            evaluator.reset(
                world_state.vehicle_by_id(self._vehicle_id), world_state, reset_time
            )
            if bool_mode:
                self.switch_to_boolean(evaluator)
            else:
                self.switch_to_robustness(evaluator)
            rule_rob = []
            other_ids = []
            while evaluator.current_time < evaluator.ego_vehicle.end_time:
                rule_rob.append(evaluator.update())
                other_ids.append(evaluator.other_ids)
                # if rule_rob[-1] < 0:
                #     break
            rule_rob_all.append(np.array(rule_rob, dtype=np.float64))
            other_ids_all.append(other_ids)
        return rule_rob_all, other_ids_all

    def query_rule_rob_all(self):
        """
        Queries the robustness value and the other vehicle id with the minimum robustness
        """
        if self.rob_rule is None:
            raise ValueError("the evaluation procedure is not executed yet")
        return self.rob_rule, self.other_ids

    def _cal_tv_def(self):
        # Check if there is an immediate violation at the first time step
        if np.any(self.rob_rule[:, 0] < 0):
            return -math.inf
        all_id_all_props_tv = dict()
        #
        # for rule_idx in range(len(self._rules)):
        #     all_id_all_props_tv[self._rules[rule_idx]] = dict()

        for prop_node in self._prop_nodes:
            rule_idx = self._rules.index(prop_node.source_rule)

            # Initialize the dictionary for the current rule if it doesn't already exist
            if self._rules[rule_idx] not in all_id_all_props_tv:
                all_id_all_props_tv[self._rules[rule_idx]] = dict()

            source_rule = prop_node.source_rule
            # index of the source rule
            source_rule_idx = self._rules.index(source_rule)

            evaluation = self.all_props_all_ids_all[source_rule_idx][prop_node.name]
            for veh in evaluation.keys():
                if veh not in all_id_all_props_tv[self._rules[rule_idx]]:
                    all_id_all_props_tv[self._rules[rule_idx]][veh] = dict()

                # Initialize lists for tv_satisfaction and tv_violation
                tv_satisfaction = []
                tv_violation = []

                # List of time evaluations for a given vehicle and proposition
                time_values = evaluation[veh]

                # Start index after initial -inf values are skipped
                start_index = None

                # Iterate over the time_values and build both lists
                for idx, value in enumerate(time_values):
                    # Skip leading -inf values
                    if value == float("-inf") and start_index is None:
                        # todo: skip the valuations to align the tv with the future/once operator
                        continue

                    # Mark the first non -inf index as the start_index
                    if start_index is None:
                        start_index = idx

                    # Handle satisfaction for positive or inf values
                    if value > 0 or value == float("inf"):
                        tv_satisfaction.append(float("inf"))
                        tv_violation.append(
                            idx - start_index - self._start_time_step
                        )  # Violation: positive values get adjusted index
                    else:
                        # Handle satisfaction for negative values
                        tv_satisfaction.append(
                            idx - start_index - self._start_time_step
                        )  # Satisfaction: negative values get adjusted index
                        tv_violation.append(
                            float("inf")
                        )  # Violation: negative values get inf

                # todo: more complicated version
                # # Exception for processing prop_node logic
                # if prop_node.name[5:6] != prop_node.name[7:8]:
                #     sub_name = prop_node.name[:5] + prop_node.name[7] + prop_node.name[6:]
                #     sub_evaluation = self.all_props_all_ids_all[rule_idx][sub_name][veh]
                #     sub_tv = next(
                #         (i for i, v in enumerate(sub_evaluation) if v != float('-inf') and v < 0),
                #         None
                #     ) # first index where the value is non-inf and < 0
                #     # fixme: sub_tv for satisfaction
                #     if sub_tv is not None:  # Make sure sub_tv is not None (valid)
                #         # Count the number of -inf values before the first valid negative value (sub_tv)
                #         len_inf = len([v for v in sub_evaluation[:sub_tv] if v == float('-inf')])
                #
                #         # Initialize tv_violation as an empty list
                #         tv_violation = []
                #
                #         # Iterate over sub_evaluation and calculate min over the slice
                #         for idx in range(len(sub_evaluation)):
                #             # Ensure the slice doesn't exceed the length of sub_evaluation
                #             slice_end = min(idx + len_inf, len(sub_evaluation))
                #
                #             # Take the slice from idx to slice_end and find the min, excluding -inf values
                #             slice_values = [v for v in sub_evaluation[idx:slice_end] if v != float('-inf')]
                #
                #             if slice_values:
                #                 # Append the minimum of the valid slice values
                #                 tv_violation.append(min(slice_values))
                #             else:
                #                 # If the slice is empty or contains only -inf, append float('inf') or another placeholder
                #                 tv_violation.append(float('inf'))

                all_id_all_props_tv[self._rules[rule_idx]][veh][
                    prop_node.alphabet
                ] = tv_satisfaction
                all_id_all_props_tv[self._rules[rule_idx]][veh][
                    "~" + prop_node.alphabet
                ] = tv_violation

        rule_to_tv = {}
        rule_to_other_id = {}
        violated_rules = []

        for idx in range(len(self._rules)):
            rule = self._rules[idx]
            tv_list = []
            tv_by_veh = {}
            sat_formula_ind = self.sat_formula_sep[rule]
            parsed_nnf_formula: NNFFormula = parse_nnf_formula(str(sat_formula_ind))

            # Calculate TV for each vehicle for this rule
            for veh, props_tv in all_id_all_props_tv[rule].items():
                if len(parsed_nnf_formula.compute_tv_list(props_tv)) == 0:
                    tv = math.inf
                else:
                    tv = min(
                        parsed_nnf_formula.compute_tv_list(props_tv)
                    )  # min: globally

                # Replace tv with inf if it's equal to start_time_step
                tv = math.inf if tv == self._start_time_step else tv

                tv_list.append(tv)
                tv_by_veh[veh] = tv

            # Find the minimum TV and the corresponding vehicle
            min_tv = min(tv_list)
            rule_to_tv[rule] = min_tv
            rule_to_other_id[rule] = next(
                veh for veh, tv in tv_by_veh.items() if tv == min_tv
            )

            # If the rule has a valid TV (i.e., finite), it is considered violated
            if min_tv != math.inf:
                violated_rules.append(rule)

        # Determine the minimum TV overall and the corresponding rule
        if violated_rules:
            min_rule_idx = min(violated_rules, key=lambda rule: rule_to_tv[rule])
            min_tv = rule_to_tv[min_rule_idx]
        else:
            min_rule_idx = None
            min_tv = math.inf

        # Convert the rule index to integer if possible
        min_rule_idx = (
            self._rules.index(min_rule_idx) if min_rule_idx is not None else None
        )

        # Return: violated rules, the index of the rule with the minimum TV, the minimum TV as an integer, rule-to-TV dictionary, and rule-to-other-ID dictionary
        return (
            violated_rules,
            min_rule_idx,
            int(min_tv) if min_tv != math.inf else math.inf,
            rule_to_tv,
            rule_to_other_id,
        )

    def _cal_tv_initial(
        self,
    ) -> Tuple[
        List[str], int, Union[int, float], Dict[str, Union[int, float]], Dict[str, Any]
    ]:
        """
        Calculate the initial time-to-violation (TV) for the monitored rules.

        This function evaluates when a rule is violated based on the robustness measure.
        It returns the index of the violated rule, the time of violation, and an associated identifier.

        Returns:
            Tuple[List[str], int, Union[int, float], Dict[str, Union[int, float]], Dict[str, Any]]:
                - List of names of all violated rules (or empty list if no violations).
                - Index of the rule with the minimum TV (or -1 if no violations).
                - Minimum time-to-violation (TV) across all rules (or inf if no violations).
                - Dictionary mapping each rule name to its corresponding TV.
                - Dictionary mapping each rule name to the corresponding other ID.
        """
        rule_to_tv = {}
        rule_to_other_id = {}
        violated_rules = []
        min_tv = float("inf")  # Initialize min_tv to infinity
        min_rule_idx = -1  # Initialize min_rule_idx to -1 (no violations)

        # Check if there is an immediate violation at the first time step
        if np.any(self.rob_rule[:, 0] < 0):
            return (
                violated_rules,
                min_rule_idx,
                -math.inf,
                rule_to_tv,
                rule_to_other_id,
            )  # all violated

        # Calculate the time-to-violation for each rule
        tv_per_rule = np.argmax(self.rob_rule < 0, axis=-1) + self._start_time_step

        # Populate rule_to_tv and rule_to_other_id dictionaries
        for idx, tv in enumerate(tv_per_rule):
            rule_name = self._rules[idx]
            # print  rule with the tv
            print(f"\tRule: {rule_name} - TV: {tv}")

            if tv == self._start_time_step:
                rule_to_tv[rule_name] = float("inf")
                rule_to_other_id[rule_name] = None
            else:
                rule_to_tv[rule_name] = tv
                if self.other_ids[idx][tv - self._start_time_step] == ():
                    rule_to_other_id[rule_name] = self._vehicle_id
                else:
                    rule_to_other_id[rule_name] = self.other_ids[idx][
                        tv - self._start_time_step
                    ][0]

                violated_rules.append(rule_name)  # Add the rule to the violated list
                if tv < min_tv:
                    min_tv = tv  # Update the minimum TV
                    min_rule_idx = (
                        idx  # Update the index of the rule with the minimum TV
                    )

        # If no violations occurred, return an empty list, -1 for index, and inf TV
        if not violated_rules:
            return [], -1, float("inf"), rule_to_tv, rule_to_other_id

        return violated_rules, min_rule_idx, int(min_tv), rule_to_tv, rule_to_other_id

    def switch_to_boolean(self, evaluator):
        if not evaluator._eval_visitor.use_boolean:
            evaluator._eval_visitor.use_boolean = True

    def switch_to_robustness(self, evaluator):
        if evaluator._eval_visitor.use_boolean:
            evaluator._eval_visitor.use_boolean = False


# Currently, MTL monitor is not supported
# class MTLRuleMonitor:
#     def __init__(self,
#                  scenario: Scenario,
#                  ego_id: int,
#                  rule_set: Union[str, Iterable[str]]):
#         self.rule_eval = CommonRoadObstacleEvaluation(os.path.dirname(__file__) + "/../../config/")
#         self.rule_eval.activated_traffic_rule_sets = rule_set
#         assert self.rule_eval.simulation_param["evaluation_mode"] == "test", "<MTLRuleMonitor>: the given evaluation " \
#                                                                              "mode {} is invalid".\
#             format(self.rule_eval.simulation_param["evaluation_mode"])
#         self.rule_eval.update_eval_dict()
#         self._scenario = scenario
#         self._ego_id = ego_id
#
#     def evaluate_initially(self):
#         """
#         Evaluate the rule violation initially - if violated, return the corresponding rule-relevant vehicle (if existed)
#         """
#         eval_result = self.rule_eval.evaluate_scenario(self._scenario)
#         ego_result = None
#         for veh_id, evaluation in eval_result:
#             if veh_id == self._ego_id:
#                 ego_result = evaluation
#                 break
#         violation_boolean = False
#         violation_veh = list()
#         for rule_str, result in ego_result.items():
#             if not result:
#                 violation_veh.append(int(rule_str[-4:]))
#                 violation_boolean = True
#         return violation_boolean, violation_veh
