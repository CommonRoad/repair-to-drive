import copy

from matplotlib import pyplot as plt

from crrepairer.smt.monitor_wrapper import STLRuleMonitor
from crrepairer.repairer.smt_repairer import SMTTrajectoryRepairer
from crrepairer.utils.configuration import RepairerConfiguration
from crrepairer.utils.repair import retrieve_ego_vehicle
from crrepairer.utils.visualization import draw_scenario

import math

# =========================================================================
# Example from the Proc. of the IEEE paper (see Fig. 7 in):
# M. Althoff, S. Maierhofer, G. Würsching, Y. Lin, F. Lercher, and R. Stolz,
# "No more traffic tickets: A tutorial to ensure traffic-rule compliance of
# automated vehicles," Proc. IEEE, pp. 1–30, early access, 2025.
#
# Violated traffic rule: R_G1 (see also):
# S. Maierhofer, A.-K. Rettinger, E. C. Mayer, and M. Althoff,
# "Formalization of interstate traffic rules in temporal logic,"
# in Proc. IEEE Intell. Vehicles Symp., 2020, pp. 752–759.
# =========================================================================

if __name__ == "__main__":
    # ========== Scenario and Configuration =========
    scenario_id = "ZAM_Augmentation-1_1_T-1"

    # Build configuration object
    config = RepairerConfiguration()

    config.general.set_path_scenario(scenario_id)
    config.update()
    config.repair.rules = ["R_G1"]
    config.repair.ego_id = 42
    config.debug.show_plots = True
    config.repair.constraint_mode = 1
    config.repair.use_mpr = False
    config.miqp_planner.slack_long = True
    config.repair.N_r = 31
    config.scenario.remove_obstacle(config.scenario.obstacle_by_id(100))
    planning_problem_original = copy.deepcopy(config.planning_problem)

    # Retrieve the ego vehicle
    ego_initial = retrieve_ego_vehicle(config)

    # Fix the other vehicle
    ego_target = config.scenario.obstacle_by_id(102)

    # Updated prediction
    for state in ego_target.prediction.trajectory.state_list:
        state.position[0] -= 2
        state.velocity -= 1

    # ========== Traffic Rule Monitor =========
    traffic_rule_monitor = STLRuleMonitor(config)

    # ========== Trajectory Repairing =========
    if traffic_rule_monitor.tv_time_step is not math.inf:
        repairer = SMTTrajectoryRepairer(traffic_rule_monitor, ego_initial, config)

        repaired_traj = repairer.repair()
        if repaired_traj is not None:
            ego_repaired = repairer.convert_traj_to_ego_vehicle(
                ego_initial.obstacle_shape, ego_initial.initial_state, repaired_traj
            )

            # ========= Visualize the results =========

            # Plot both on vertically stacked subplots
            fig, axs = plt.subplots(2, 1, figsize=(20, 16), sharex=True)

            # Draw scenario with violations
            draw_scenario(
                axs[0],
                config.scenario,
                planning_problem_original,
                config.repair.N_r - 1,
                ego_initial,
                repairer.tv,
                repairer.tc,
                y_shift=0.0,
                title="Violation",
                flag_violation=True,
            )

            # Draw scenario with repair
            config.scenario.remove_obstacle(
                config.scenario.obstacle_by_id(config.repair.ego_id)
            )
            config.scenario.add_objects(ego_repaired)
            draw_scenario(
                axs[1],
                config.scenario,
                planning_problem_original,
                config.repair.N_r - 1,
                ego_repaired,
                repairer.tv,
                repairer.tc,
                y_shift=0.0,
                title="Repaired",
            )

            for ax in axs:
                ax.axis("off")

            plt.show()

            # =========  Evaluate the repaired trajectory =========
            config.scenario.assign_obstacles_to_lanelets(
                obstacle_ids={ego_repaired.obstacle_id}
            )
            config.repair.ego_id = ego_repaired.obstacle_id

            traffic_rule_monitor = STLRuleMonitor(config)
