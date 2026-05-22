import os
import sys
import numpy as np

from .simulation.sim_loop import run_simulation, TRAJECTORIES
from .plotting import plot_runs, plot_runs_paper

CONDITIONS = ["Nominal", "Wind Only", "Payload Only", "Wind + Payload"]
CONTROLLERS = ["PID", "MPC", "PID+MPC"]

def choose_option(title, options):
    print(f"\nSelect {title}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        try:
            choice = int(input("Enter choice (number): ").strip())
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except ValueError:
            pass
        print(f"Invalid selection. Please enter a number between 1 and {len(options)}.")

def run_single_interactive():
    print("\n--- Single Simulation Setup ---")
    traj_name = choose_option("trajectory", list(TRAJECTORIES.keys()))
    cond_name = choose_option("condition", CONDITIONS)
    ctrl_choice = choose_option("controller", CONTROLLERS + ["Compare All"])
    
    gui_input = input("Run with GUI? (y/n, default: y): ").strip().lower()
    gui = gui_input != 'n'

    selected_controllers = CONTROLLERS if ctrl_choice == "Compare All" else [ctrl_choice]
    
    runs = []
    for ctrl in selected_controllers:
        print(f"\nRunning {ctrl} on {traj_name} under {cond_name}...")
        res = run_simulation(traj_name, cond_name, ctrl, gui=gui)
        runs.append((res["controller"], res["ref_traj"], res["actual_traj"], res["traj_name"], res["condition"]))
        
    print("\nSimulation(s) completed. Generating plots...")
    plot_runs(runs, auto_save=True)
    plot_runs_paper(runs, auto_save=True)

def run_batch_experiments():
    print("\n--- Running Headless Batch Experiments ---")
    raw_data_dir = os.path.join("experiment_results", "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    total_runs = len(TRAJECTORIES) * len(CONDITIONS) * len(CONTROLLERS)
    print(f"Saving raw data (.npz) to: {raw_data_dir}")
    print(f"Total simulations to run: {total_runs}")
    print("-" * 60)
    
    run_idx = 1
    for traj_name in TRAJECTORIES.keys():
        for cond_name in CONDITIONS:
            comparison_runs = []
            for ctrl_name in CONTROLLERS:
                print(f"[{run_idx}/{total_runs}] Running: {ctrl_name} | {traj_name} | {cond_name}...", end=" ", flush=True)
                
                try:
                    res = run_simulation(traj_name, cond_name, ctrl_name, gui=False, save_dir=os.path.join("npy_data", "batch"))
                    
                    # Save in legacy batch format (.npz) for backward compatibility
                    legacy_results = {
                        "t": np.array(res["times"]),
                        "pos": res["actual_traj"],
                        "ref_pos": res["ref_traj"],
                        "error": np.array(res["errors"]),
                        "rpm": np.array(res["rpms"])
                    }
                    
                    safe_name = f"{ctrl_name}_{traj_name}_{cond_name}".replace(" ", "_").replace("+", "plus")
                    save_path = os.path.join(raw_data_dir, f"{safe_name}.npz")
                    np.savez(save_path, **legacy_results)
                    
                    comparison_runs.append((res["controller"], res["ref_traj"], res["actual_traj"], res["traj_name"], res["condition"]))
                    print("SUCCESS")
                except Exception as e:
                    print(f"FAILED: {e}")
                
                run_idx += 1
            
            # Generate and save comparison plots for this trajectory + condition combination
            if len(comparison_runs) == len(CONTROLLERS):
                try:
                    plot_runs(comparison_runs, auto_save=True)
                    plot_runs_paper(comparison_runs, auto_save=True)
                except Exception as e:
                    print(f"  [Plotting failed: {e}]")

    print("-" * 60)
    print("Batch experiments completed successfully!")

def main():
    print("=" * 60)
    print("  Adaptive Hybrid Drone Control Simulation Package")
    print("=" * 60)
    
    menu = [
        "Run Single Simulation (Interactive)",
        "Run Batch Experiments (All combinations, Headless)",
        "Exit"
    ]
    
    choice = choose_option("operation mode", menu)
    
    if choice == menu[0]:
        run_single_interactive()
    elif choice == menu[1]:
        run_batch_experiments()
    else:
        print("Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
