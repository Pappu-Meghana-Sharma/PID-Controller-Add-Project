import os
import time
import numpy as np
import pybullet as p
from datetime import datetime

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from ..config import (
    BASE_MASS, PAYLOAD_MASS, DURATION, CTRL_FREQ, CTRL_DT, GRAVITY,
    MIN_RPM, MAX_RPM, HOVER_RPM_BASE, INIT_RPYS,
    WIND_NONE, WIND_LIGHT, WIND_MEDIUM, PRINT_HZ
)
from ..traj import traj_hover, traj_circle, traj_figure8
from ..controllers.mpc import LinearOuterMPC, accel_to_attitude_thrust
from ..controllers.adaptive import LyapunovMonitor, AdaptiveScheduler, AdaptiveGainScheduler
from .env_setup import initialize_env, add_payload_sphere

TRAJECTORIES = {
    "Hover":    traj_hover,
    "Circle":   traj_circle,
    "Figure-8": traj_figure8,
}

def save_results(traj_name, condition_name, controller_name,
                 errors, ctrl_efforts, step_times, wall_total,
                 lyap_data=None):
    sim_total    = len(errors) * CTRL_DT
    rt_factor    = sim_total / wall_total if wall_total > 0 else 0.0
    errors       = np.array(errors)
    ctrl_efforts = np.array(ctrl_efforts)
    step_times   = np.array(step_times)

    print("\n  " + "=" * 62)
    print(f"  Controller  : {controller_name}  |  Traj: {traj_name}  |  {condition_name}")
    print(f"  Sim: {sim_total:.1f}s  |  Wall: {wall_total:.1f}s  |  RT: {rt_factor:.2f}x")
    print("  --- Tracking error (m) ---")
    print(f"    RMSE    : {np.sqrt(np.mean(errors**2)):.4f}")
    print(f"    MAE     : {np.mean(errors):.4f}")
    print(f"    Max     : {np.max(errors):.4f}")
    print(f"    Steady  : {np.mean(errors[len(errors)//2:]):.4f}  (2nd half)")
    print("  --- Control effort ---")
    print(f"    Mean RPM²: {np.mean(ctrl_efforts):.1f}")
    print(f"    Max  RPM²: {np.max(ctrl_efforts):.1f}")
    print("  --- Timing ---")
    print(f"    Mean step: {np.mean(step_times):.1f} ms  (budget {CTRL_DT*1000:.1f} ms)")
    print(f"    Over budget: {np.sum(step_times > CTRL_DT*1000)}/{len(step_times)}")
    if lyap_data is not None:
        V_hist = np.array(lyap_data["V"])
        s_hist = np.array(lyap_data["sigma"])
        hz_hist = np.array(lyap_data.get("mpc_hz", []))
        print("  --- Lyapunov (adaptive) ---")
        print(f"    Mean V    : {np.mean(V_hist):.4f}")
        print(f"    Mean sigma: {np.mean(s_hist):.4f}")
        print(f"    Min  sigma: {np.min(s_hist):.4f}")
        if len(hz_hist) > 0:
            print(f"    MPC Hz    : {np.mean(hz_hist):.1f} avg  "
                  f"[{np.min(hz_hist):.0f}–{np.max(hz_hist):.0f}]")
    print("  " + "=" * 62)

    line = (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] "
        f"{controller_name:<8} | {traj_name:<9} | {condition_name:<16} | "
        f"RMSE={np.sqrt(np.mean(errors**2)):.4f} | "
        f"MAE={np.mean(errors):.4f} | "
        f"MaxErr={np.max(errors):.4f} | "
        f"Steady={np.mean(errors[len(errors)//2:]):.4f} | "
        f"MeanEffort={np.mean(ctrl_efforts):.1f} | "
        f"RT={rt_factor:.2f}x | "
        f"MeanStep={np.mean(step_times):.1f}ms\n"
    )
    with open("results.txt", "a") as f:
        f.write(line)
    print("  Appended to results.txt")

def run_simulation(traj_name, condition_name, controller_name, gui=True, save_dir=None):
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        save_dir = os.path.join("npy_data", "linear", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    traj_fn = TRAJECTORIES[traj_name]

    conditions = {
        "Nominal":        {"total_mass": BASE_MASS,               "wind": WIND_NONE},
        "Wind Only":      {"total_mass": BASE_MASS,               "wind": WIND_LIGHT},
        "Payload Only":   {"total_mass": BASE_MASS + PAYLOAD_MASS, "wind": WIND_NONE},
        "Wind + Payload": {"total_mass": BASE_MASS + PAYLOAD_MASS, "wind": WIND_MEDIUM},
    }
    cfg         = conditions[condition_name]
    total_mass  = cfg["total_mass"]
    wind        = cfg["wind"]
    has_payload = total_mass > BASE_MASS

    print(f"\n  Trajectory  : {traj_name}")
    print(f"  Condition   : {condition_name}")
    print(f"  Controller  : {controller_name}")
    print(f"  Total mass  : {total_mass*1000:.1f} g")

    # Initialize environment
    env, pyb_client, obs, _ = initialize_env(gui=gui)

    # Setup controller
    pid_ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

    use_mpc    = controller_name in ("MPC", "PID+MPC")
    lyap_mon   = None
    scheduler  = None
    gain_sched = None

    if use_mpc:
        print("  Building LinearOuterMPC (CVXPY/OSQP)...")
        # Ensure we start with NOMINAL MPC frequency
        from ..config import MPC_EVERY_NOM
        mpc_dt = MPC_EVERY_NOM * CTRL_DT
        mpc    = LinearOuterMPC(dt=mpc_dt, horizon=20, mass=BASE_MASS)
        print("  MPC ready.")

        if controller_name == "PID+MPC":
            lyap_mon   = LyapunovMonitor(mpc.P_matrix)
            scheduler  = AdaptiveScheduler()
            gain_sched = AdaptiveGainScheduler(pid_ctrl)
            print("  Lyapunov monitor + adaptive scheduler + gain scheduler ready.")

    if has_payload:
        add_payload_sphere(pyb_client, env.DRONE_IDS[0], PAYLOAD_MASS)
        print("  Payload sphere attached (red, r=15 mm)")

    # Accumulators
    errors, rpms, ctrl_efforts, step_times, sim_times = [], [], [], [], []
    actual_traj, ref_traj = [], []
    lyap_log = {"V": [], "sigma": [], "mpc_hz": [], "gain_scale": []}

    action         = np.zeros((1, 4))
    N_STEPS        = int(DURATION * CTRL_FREQ)
    print_interval = max(1, CTRL_FREQ // PRINT_HZ)

    mpc_counter      = 0
    from ..config import MPC_EVERY_NOM
    mpc_every        = MPC_EVERY_NOM
    last_rpy_cmd     = INIT_RPYS[0].copy()
    last_F_total     = BASE_MASS * GRAVITY
    last_pure_mpc_rpm = np.full(4, HOVER_RPM_BASE)

    print(f"\n  Flying {DURATION:.0f}s")
    print(f"  {'t':>6}  {'Err(m)':>8}  {'V':>10}  {'sigma':>8}  {'MPC Hz':>7}  {'GScale':>7}  {'Status':>8}")
    print("  " + "-" * 70)

    wall_start = time.time()

    for i in range(N_STEPS):
        step_wall_start = time.time()
        t = i * CTRL_DT

        state   = obs[0]
        cur_pos = state[0:3].copy()
        cur_vel = state[10:13].copy()

        target_pos, target_vel = traj_fn(t)
        actual_traj.append(cur_pos.copy())
        ref_traj.append(target_pos.copy())

        # Lyapunov update
        sigma      = 0.0
        gain_scale = 1.0
        if lyap_mon is not None:
            sigma = lyap_mon.update(cur_pos, cur_vel, target_pos, target_vel, t)

        # Adaptive scheduling
        if scheduler is not None:
            mpc_every = scheduler.update(sigma)

        # Adaptive gains
        if gain_sched is not None:
            pos_err_norm = float(np.linalg.norm(cur_pos - target_pos))
            gain_scale   = gain_sched.update(sigma, pos_err_norm)

        # Controller execution
        if controller_name == "PID":
            rpm, _, _ = pid_ctrl.computeControlFromState(
                control_timestep=CTRL_DT, state=state,
                target_pos=target_pos, target_vel=target_vel,
                target_rpy=INIT_RPYS[0])
            rpm = np.clip(rpm, MIN_RPM, MAX_RPM)

        elif controller_name == "MPC":
            if mpc_counter % MPC_EVERY_NOM == 0:
                acc_cmd = mpc.compute(cur_pos, cur_vel, traj_fn, t)
                target_rpy, F_total = accel_to_attitude_thrust(acc_cmd, BASE_MASS, state[9])
                base_rpm   = np.sqrt(F_total / (4.0 * pid_ctrl.KF))
                pwm_thrust = (base_rpm - pid_ctrl.PWM2RPM_CONST) / pid_ctrl.PWM2RPM_SCALE
                att_out    = pid_ctrl._dslPIDAttitudeControl(
                    CTRL_DT, pwm_thrust, state[3:7], target_rpy, np.zeros(3))
                rpm = att_out[0] if isinstance(att_out, tuple) else att_out
                rpm = np.clip(rpm, MIN_RPM, MAX_RPM)
                last_pure_mpc_rpm = rpm.copy()
            else:
                rpm = last_pure_mpc_rpm.copy()
            mpc_counter += 1

        elif controller_name == "PID+MPC":
            if mpc_counter % mpc_every == 0:
                new_acc_cmd = mpc.compute(cur_pos, cur_vel, traj_fn, t)
                last_rpy_cmd, last_F_total = accel_to_attitude_thrust(
                    new_acc_cmd, BASE_MASS, state[9])

            mpc_counter += 1

            base_rpm   = np.sqrt(last_F_total / (4.0 * pid_ctrl.KF))
            pwm_thrust = (base_rpm - pid_ctrl.PWM2RPM_CONST) / pid_ctrl.PWM2RPM_SCALE
            att_out    = pid_ctrl._dslPIDAttitudeControl(
                CTRL_DT, pwm_thrust, state[3:7], last_rpy_cmd, np.zeros(3))
            rpm = att_out[0] if isinstance(att_out, tuple) else att_out
            rpm = np.clip(rpm, MIN_RPM, MAX_RPM)
        else:
            raise ValueError(f"Unknown controller: {controller_name}")

        action[0, :] = rpm

        # Apply disturbances and step simulation
        try:
            if np.any(wind != 0):
                wind_scale = 1.0 + 0.4 * np.sin(1.2 * t) + 0.2 * np.sin(3.5 * t)
                w_varying  = wind * wind_scale
                w_noisy    = w_varying + np.random.normal(0, 0.05 * np.linalg.norm(wind), 3)
                p.applyExternalForce(
                    env.DRONE_IDS[0], -1,
                    w_noisy.tolist(), [0.0, 0.0, 0.0],
                    p.WORLD_FRAME, physicsClientId=pyb_client)
            obs, _, terminated, truncated, _ = env.step(action)
        except Exception as e:
            print(f"  [!] Simulation error: {e}")
            break

        if terminated or truncated:
            print("  [!] Episode ended early.")
            break

        # Metrics collection
        err      = float(np.linalg.norm(cur_pos - target_pos))
        effort   = float(np.sum((rpm / 1000.0) ** 2))
        step_ms  = (time.time() - step_wall_start) * 1000.0

        errors.append(err)
        rpms.append(float(np.mean(rpm)))
        ctrl_efforts.append(effort)
        step_times.append(step_ms)
        sim_times.append(t)

        if lyap_mon is not None:
            lyap_log["V"].append(lyap_mon.V)
            lyap_log["sigma"].append(lyap_mon.sigma)
            lyap_log["mpc_hz"].append(scheduler.current_mpc_hz if scheduler else MPC_FREQ_NOM)
            lyap_log["gain_scale"].append(gain_scale)

        if i % print_interval == 0:
            mpc_hz_str  = f"{scheduler.current_mpc_hz:.0f}" if scheduler else " N/A"
            gs_str      = f"{gain_scale:.2f}" if gain_sched else " N/A"
            V_str       = f"{lyap_mon.V:.3f}" if lyap_mon else "    N/A"
            sigma_str   = f"{sigma:+.3f}" if lyap_mon else "   N/A"
            status      = lyap_mon.stability_label if lyap_mon else "   N/A"
            print(f"  {t:6.2f}s  {err:>8.4f}  {V_str:>10}  {sigma_str:>8}  "
                  f"{mpc_hz_str:>7}  {gs_str:>7}  {status:>8}")

        elapsed = time.time() - step_wall_start
        if gui and CTRL_DT - elapsed > 0:
            time.sleep(CTRL_DT - elapsed)

    wall_total = time.time() - wall_start

    try:
        env.close()
    except Exception:
        pass

    # Save results to disk
    safe = f"{controller_name}_{traj_name}_{condition_name}".replace(" ", "_").replace("+", "plus")
    np.save(os.path.join(save_dir, f"traj_{safe}_actual.npy"), np.array(actual_traj))
    np.save(os.path.join(save_dir, f"traj_{safe}_target.npy"), np.array(ref_traj))  # Standardized suffix

    if lyap_mon is not None:
        np.save(os.path.join(save_dir, f"lyap_{safe}.npy"), {
            "V":          np.array(lyap_log["V"]),
            "sigma":      np.array(lyap_log["sigma"]),
            "mpc_hz":     np.array(lyap_log["mpc_hz"]),
            "gain_scale": np.array(lyap_log["gain_scale"]),
        })

    save_results(traj_name, condition_name, controller_name,
                 errors, ctrl_efforts, step_times, wall_total,
                 lyap_data=lyap_log if lyap_mon else None)

    return {
        "controller":   controller_name,
        "traj_name":    traj_name,
        "condition":    condition_name,
        "times":        sim_times,
        "errors":       errors,
        "rpms":         rpms,
        "ctrl_efforts": ctrl_efforts,
        "actual_traj":  np.array(actual_traj),
        "ref_traj":     np.array(ref_traj),
        "lyap_log":     lyap_log,
        "rmse":         float(np.sqrt(np.mean(np.array(errors)**2))),
    }
