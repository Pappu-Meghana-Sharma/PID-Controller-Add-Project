import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import pandas as pd
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
import time
from gym_pybullet_drones.utils.utils import sync

def verify_with_plots():
    SCENARIOS = [
        [0.027, 0.000, 0.000, "Ideal (Baseline)"],
        [0.030, 0.003, 0.000, "Medium (Droop/Drift)"],   # wind only in X
        [0.035, 0.004, 0.000, "Heavy (Max Stress)"]      # wind only in X
    ]
    DURATION_SEC = 20
    GUI = True

    all_data = []   # will hold dicts with Time, X, Y, Z, Scenario

    for mass, wx, wy, label in SCENARIOS:
        print(f"\n>>> Running {label} ...")

        env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            initial_xyzs=np.array([[0, 0, 0.7]]),
            physics=Physics.PYB_GND_DRAG_DW,
            gui=GUI
        )
        client = env.getPyBulletClient()
        ctrl = DSLPIDControl(DroneModel.CF2X)
        p.changeDynamics(env.DRONE_IDS[0], -1, mass=mass, physicsClientId=client)

        obs, _ = env.reset()
        dt = env.CTRL_TIMESTEP
        steps = int(DURATION_SEC * env.CTRL_FREQ)

        start_time = time.time()
        target_z_history = []   # to store target Z for plotting

        for step in range(steps):
            t = step * dt

            # ---- Target trajectory (same as final script) ----
            if t < 5:
                target_pos = np.array([0, 0, 1.0])
                target_vel = np.zeros(3)
            elif t < 12:
                target_pos = np.array([0.0, 0.3, 0.9])
                target_vel = np.zeros(3)
            else:
                t_adj = t - 12
                r, w = 0.6, 0.8
                target_pos = np.array([r * np.cos(w * t_adj), r * np.sin(w * t_adj), 0.8])
                target_vel = np.array([-r * w * np.sin(w * t_adj), r * w * np.cos(w * t_adj), 0])

            target_z_history.append(target_pos[2])

            # ---- Compute action ----
            action_raw, _, _ = ctrl.computeControlFromState(
                control_timestep=dt,
                state=obs[0],
                target_pos=target_pos,
                target_vel=target_vel
            )
            action = action_raw.flatten()

            # ---- Apply wind ----
            p.applyExternalForce(env.DRONE_IDS[0], -1, [wx, wy, 0], [0,0,0],
                                 p.WORLD_FRAME, physicsClientId=client)

            # ---- Step simulation ----
            obs, _, _, _, _ = env.step(action.reshape(1,4))

            # ---- Record true position ----
            true_pos = obs[0][0:3]
            all_data.append({
                'Time': t,
                'X': true_pos[0],
                'Y': true_pos[1],
                'Z': true_pos[2],
                'Scenario': label
            })
    
            # ---- Sync for GUI ----
            if GUI:
                sync(step, start_time, dt)

            # ---- Safety stop ----
            if true_pos[2] < 0.05:
                print(f"   Drone grounded at t={t:.2f}s")
                break

        env.close()

    # ---- Convert to DataFrame ----
    df = pd.DataFrame(all_data)

    # ---- Dual visualisation with correct target lines ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Altitude (Z) with target Z over time
    for label in df['Scenario'].unique():
        sub = df[df['Scenario'] == label]
        ax1.plot(sub['Time'], sub['Z'], label=label, linewidth=2)
    # Add target Z as a dashed black line (using the first scenario's target, which is the same for all)
    # Since target is same for all scenarios, we can compute it once
    t_vals = np.linspace(0, DURATION_SEC, 1000)
    target_z_vals = []
    for tt in t_vals:
        if tt < 5:
            target_z_vals.append(1.0)
        elif tt < 12:
            target_z_vals.append(0.9)
        else:
            target_z_vals.append(0.8)
    ax1.plot(t_vals, target_z_vals, 'k--', label='Target Z', linewidth=2)
    ax1.set_title("Altitude (Z) Analysis: Steady-State Droop")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Altitude (m)")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Top-Down Drift (X vs Y)
    for label in df['Scenario'].unique():
        sub = df[df['Scenario'] == label]
        if "Ideal" in label:
            ax2.plot(sub['X'], sub['Y'], color='magenta', linewidth=4,
                     linestyle='--', label=label, zorder=10)
            ax2.scatter(0, 0, color='magenta', s=100, edgecolors='black', zorder=11)
        else:
            ax2.plot(sub['X'], sub['Y'], linewidth=2, label=label)
    ax2.scatter(0, 0, color='red', marker='x', s=150, label='Target (0,0)', zorder=12)
    ax2.set_title("Horizontal (XY) Drift Analysis: Wind Impact")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_with_plots()