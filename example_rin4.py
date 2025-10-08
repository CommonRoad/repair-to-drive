from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.visualization import visualize_repaired_result, visualize_scenario_once
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

import matplotlib
matplotlib.use('TkAgg')

import math

# ==============================================================
# Violated traffic rule: R_IN4 (see):
# S. Maierhofer, P. Moosbrugger, and M. Althoff, “Formalization of
# intersection traffic rules in temporal logic,” in Proc. of the
# IEEE Intell. Vehicles Symp., 2022, pp. 1135–1144.
# ==============================================================

if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario_id = "DEU_AachenBendplatz-1_162280_T-2299"

    # Build configuration object
    config = RepairerConfiguration()
    config.general.set_path_scenario(scenario_id)
    config.update()
    config.repair.scenario_type = "intersection"
    config.repair.intersection_type = "dataset"
    config.repair.rules = ["R_IN4"]
    config.repair.ego_id = 10179

    config.repair.N_r = 20

    config.debug.show_plots = True
    config.repair.planner = 3
    config.repair.constraint_mode = 2
    config.repair.use_mpr = False
    config.repair.use_mpr_derivative = False
    config.debug.plot_limits = [40, 69, -45, -17]

    ego_initial = retrieve_ego_vehicle(config)

    # ========== Traffic Rule Monitor =========
    traffic_rule_monitor = STLRuleMonitor(config)

    # ========== Trajectory Repairing =========
    if traffic_rule_monitor.tv_time_step is not math.inf:
        repairer = SMTTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)
        repaired_traj = repairer.repair()
        if repaired_traj is not None and config.debug.show_plots:
            ego_repaired = repairer.convert_traj_to_ego_vehicle(
                ego_initial.obstacle_shape, ego_initial.initial_state, repaired_traj
            )
            visualize_repaired_result(config, ego_initial, ego_repaired, repairer)
