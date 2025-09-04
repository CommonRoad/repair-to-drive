from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.visualization import visualize_repaired_result
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

import math

# ==============================================================
# Violated traffic rule: R_IN1 (see):
# S. Maierhofer, P. Moosbrugger, and M. Althoff, “Formalization of
# intersection traffic rules in temporal logic,” in Proc. of the
# IEEE Intell. Vehicles Symp., 2022, pp. 1135–1144.
# ==============================================================

if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario_id = "DEU_TestRIN1-3_1_T-1"

    # Build configuration object
    config = RepairerConfiguration()

    config.general.set_path_scenario(scenario_id)
    config.update()

    config.repair.ego_id = 31
    config.repair.rules = ["R_IN1"]
    config.repair.scenario_type = "intersection"
    config.repair.intersection_type = "hand_draft"
    config.debug.plot_limits = [-5, 50, -4.5, 3]
    config.debug.show_plots = True
    config.miqp_planner.slack_long = False
    config.repair.N_r = 41
    config.repair.constraint_mode = 1
    config.repair.use_mpr = False

    # Retrieve the ego vehicle
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
            if config.debug.show_plots:
                # ============= Visualization =============
                visualize_repaired_result(config, ego_initial, ego_repaired, repairer)
