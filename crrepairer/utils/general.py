from typing import Tuple
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.scenario import Scenario

# commonroad-curvilinear-coordinatesystem
import commonroad_clcs.pycrccosy as pycrccosy


def load_scenario_and_planning_problem(
    path_scenario, idx_planning_problem: int = 0
) -> Tuple[Scenario, PlanningProblem, PlanningProblemSet]:
    """
    Loads a scenario and planning problem from the configuration.
    :param path_scenario: full path to scenario XML file
    :param idx_planning_problem: index of the planning problem
    :return: scenario and planning problem and planning problem set
    """
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open(True)
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[
        idx_planning_problem
    ]

    return scenario, planning_problem, planning_problem_set


def create_curvilinear_coordinate_system(
    reference_path: np.ndarray,
) -> pycrccosy.CurvilinearCoordinateSystem:
    cosy = pycrccosy.CurvilinearCoordinateSystem(reference_path)
    return cosy
