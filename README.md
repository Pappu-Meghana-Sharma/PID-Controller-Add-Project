# PID-MPC Adaptive Hybrid Drone Control Simulation

This directory contains the simulation codebase, experimental scripts, progress reports, and analysis figures for the Lyapunov-guided adaptive hybrid control system of a quadrotor drone.

## 📂 Repository Structure

The active files pushed to GitHub are organized as follows:

### 1. ⚙️ Core Modular Package (`adaptive_hybrid_control/`)
A fully modularized Python package containing clean, refactored components of the simulation:
* `config.py`: Central configurations containing drone physical parameters, frequency ranges, and simulation settings.
* `traj.py`: Trajectory profiles (`Hover`, `Circle`, `Figure-8` / Lemniscate of Bernoulli).
* `controllers/mpc.py`: Outer-loop Linear MPC state-space tracking controller implemented using `cvxpy`/`osqp`.
* `controllers/adaptive.py`: Lyapunov monitor, adaptive frequency scheduler (12Hz - 48Hz), and adaptive inner-loop PID gain scaler.
* `simulation/env_setup.py`: Environment setup, drone physics reset, payload initialization, and wind force models using PyBullet.
* `simulation/sim_loop.py`: Master simulation execution loop supporting both GUI-rendering and headless simulation.
* `plotting.py`: Academic white-theme and standard dark-theme plotting scripts.
* `main.py`: Interactive command-line runner to execute single simulation runs or batch experiments.

### 2. 📊 Analysis & Visualizations (`figures/`)
Contains regenerated comparative analysis plots of PID, baseline MPC, and the proposed PID+MPC:
* `trajectory_Circle.png` & `trajectory_Figure88.png`: 3D and 2D spatial tracking profiles.
* `error_time_Hover.png`, `error_time_Circle.png`, & `error_time_Figure88.png`: Real-time Euclidean tracking errors.
* `barplot_Hover.png`, `barplot_Circle.png`, & `barplot_Figure88.png`: Tracking performance metric comparisons (RMSE, MAE, Max Error, Steady-State Error).
* `heatmap_rmse.png`: Comprehensive performance comparison under varying wind and payload configurations.
* `summary_table.txt`: Tabular performance comparison showing RMSE, MAE, Max Error, Control Effort (RPM), and Real-Time factors.

### 3. 🧪 Experimental Baseline Scripts (`experimental_scripts/`)
Legacy, alternative configurations, and batch processing scripts used during development:
* `Drone.py` & `Drone_new.py`: Legacy monolithic implementations of Linear MPC and Nonlinear MPC (using `do-mpc`).
* `visualization.py` & `visualization_experiments.py`: Baseline non-adaptive simulation scripts.
* `run_batch.py` & `batch_run.py`: Multi-scenario batch runners.
* `simulation_core.py`: Extract of headless simulation runner.
* `test_load_stats.py`, `organize_data.py`, `generate_paper_assets.py`, `generate_analysis_plots.py`, `generate_all_plots.py`: Verification, utility, and plotting scripts.

### 4. 📝 Progress Reports & Summaries
* `Evaluation_1.pdf` & `Results_Final_Report.pdf`: Comprehensive progress reports outlining project math and experimental findings.
* `S20230030400_AddProj.pdf` & `S20230030400_add_proj_final_eval.pptx`: Project submission document and final evaluation slides.
* `professor_summary.md` & `professor_summary_2.md`: Structured markdown briefs compiling performance comparisons and key accomplishments.
* `results.txt`: Appended run-logs recording tracking error metrics for every completed simulation test.

---

## 📈 Guide to Analyzing the Figures

When presenting the generated figures to show progress, pay close attention to the following aspects:

### 1. Spatial Tracking (`trajectory_*.png`)
* **3D View**: Observe how the **PID+MPC** (green curve) matches the target trajectory (red dashed line) far more tightly, especially during rapid height changes, compared to legacy PID (blue curve) which drifts outward.
* **Top-Down View (XY-Plane)**: Look at the trajectory corners (e.g., in Figure-8). The proposed **PID+MPC** controller shows almost zero corner cutting or overshoot, while baseline MPC (orange curve) suffers from corner-shaving due to fixed execution rates, and PID shows significant drift under wind.

### 2. Temporal Tracking Error (`error_time_*.png`)
* **Initial Transient**: Note the rapid initial stabilization. The adaptive controller leverages high MPC frequency (up to 48Hz) and scaled gains to suppress initial offset in under 1.5 seconds.
* **Steady-State Bounds**: Under wind gusts and payloads, the **PID+MPC** maintains tracking errors within a tight envelope (< 1.5 cm for Hover), whereas PID and fixed-rate MPC exhibit sustained oscillations or offsets.
* **Adaptive Rate Indication**: The background shading or execution rates reveal the adaptive scheduler at work: when stability metrics decay ($V$ increases), the scheduler ramps up the outer-loop MPC execution frequency, dampening the error before returning to a power-saving low frequency (12Hz).

### 3. Metric Barplots (`barplot_*.png`)
* **Tracking Error Reduction**: Compare the height of the green (PID+MPC) bars against the orange (MPC) and blue (PID) bars. RMSE and Max Error are typically reduced by **60% to 80%** compared to PID, and **20% to 40%** compared to fixed-rate MPC.
* **Control Effort**: Verify that despite the vastly superior tracking performance, the average rotor effort (Mean RPM) remains comparable to or lower than the baselines, showing that the adaptive frequency scheme achieves performance without excessive control energy.

### 4. Robustness Heatmap (`heatmap_rmse.png`)
* **Grid Layout**: Represents RMSE across 4 environmental conditions (Nominal, Wind Only, Payload Only, Wind+Payload) and 3 trajectories (Hover, Circle, Figure-8).
* **Grid Coloring**: Cooler colors (blue/green) represent low RMSE (high tracking precision), while warmer colors (yellow/red) represent high RMSE. Note how the **PID+MPC** column remains consistently dark green/blue across all scenarios, proving its exceptional robustness to combined payload and wind disturbances.

---

## 🚀 How to Run the Code

To run the simulation locally using your environment:

1. **Start the Simulator**:
   ```bash
   python -m adaptive_hybrid_control.main
   ```
2. **Choose Mode**:
   * Select `1` for **Single Simulation (Interactive)** to choose a trajectory, condition, and controller, and view the tracking live.
   * Select `2` for **Run Batch Experiments** to run all combinations headlessly, regenerate raw data `.npz` arrays under `experiment_results/raw_data/`, and refresh the comparison figures.
