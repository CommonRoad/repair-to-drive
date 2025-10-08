from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.visualization import visualize_repaired_result
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle

import math

# ==============================================================
# Violated traffic rule: R_G1 (see):
# S. Maierhofer, A.-K. Rettinger, E. C. Mayer, and M. Althoff,
# "Formalization of interstate traffic rules in temporal logic,"
# in Proc. IEEE Intell. Vehicles Symp., 2020, pp. 752–759.
# ==============================================================

if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario_id = "DEU_Gar-1_1_T-1"

    # Build configuration object
    config = RepairerConfiguration()

    config.general.set_path_scenario(scenario_id)
    config.update()

    config.repair.ego_id = 200
    config.repair.rules = ["R_G1", "R_G3"]
    config.debug.plot_limits = [-5, 50, -4.5, 3]
    config.debug.show_plots = True
    config.miqp_planner.slack_long = True
    config.repair.N_r = 21
    config.repair.constraint_mode = 2
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
