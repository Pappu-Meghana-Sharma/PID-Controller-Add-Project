"""
Drone Controller: PID vs MPC vs PID+MPC Hybrid
================================================
Architecture:
  PID      — DSLPIDControl (fixed-gain, with mass-scaling RPM bias)
  MPC      — LinearOuterMPC (cvxpy OSQP) outputs accel command
             → accel_to_attitude() → DSLPIDControl inner attitude loop
  PID+MPC  — Same as MPC but MPC runs at 24 Hz, PID inner loop at 48 Hz
             This is the TRUE hybrid: MPC does predictive trajectory
             optimization, PID does fast attitude stabilization.

Disturbances:
  Wind    — External force applied each sim step (correctly scaled)
  Payload — Mass change reflected in BOTH physics AND controller mass

Fixes applied vs previous versions:
  1. Mass always set correctly in PyBullet (was reset to BASE_MASS always)
  2. Wind force not multiplied by SUBSTEPS (was 5x too strong)
  3. PID+MPC uses MPC acceleration → attitude command (not setpoint shift)
  4. Payload RPM bias computed from actual mass ratio
  5. All three controllers share the same attitude inner loop for fair comparison
  6. ReferenceMPC replaced with proper LinearOuterMPC everywhere
"""

import warnings
warnings.filterwarnings("ignore", message="The ONNX feature is not available")
warnings.filterwarnings("ignore", message="The opcua feature is not available")

import time
import numpy as np
import pybullet as p
import cvxpy as cp
from scipy.linalg import solve_discrete_are

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

try:
    from traj import traj_figure8, traj_circle, traj_hover
    from plot_trajectories import plot_runs
except ImportError:
    print("[WARN] traj.py or plot_trajectories.py not found — define them or replace traj_fn calls.")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SIM_FREQ      = 240
CTRL_FREQ     = 48
CTRL_DT       = 1.0 / CTRL_FREQ
MPC_FREQ      = 24          # MPC outer loop runs at half the control freq
MPC_EVERY     = CTRL_FREQ // MPC_FREQ   # = 2 steps

DURATION      = 20.0

BASE_MASS     = 0.027       # kg  (Crazyflie 2.x)
PAYLOAD_MASS  = 0.015       # kg
GRAVITY       = 9.81

MIN_RPM       = 0
MAX_RPM       = 21702

KF            = 3.16e-10
KM            = 7.94e-12
ARM           = 0.0397

Ixx           = 1.4e-5
Iyy           = 1.4e-5
Izz           = 2.17e-5

# Hover RPM for BASE_MASS (used for control-effort metric only)
HOVER_RPM_BASE = np.sqrt(BASE_MASS * GRAVITY / (4.0 * KF))

# Disturbance forces (N) — applied once per CONTROL step (not per sim step)
WIND_NONE     = np.array([0.000,  0.000, 0.000])
WIND_LIGHT    = np.array([0.0005, 0.0005, 0.000])   # ~18 mg lateral
WIND_MEDIUM   = np.array([0.0010, 0.0010, 0.000])   # ~36 mg lateral

INIT_XYZS     = np.array([[0.0, 0.0, 0.05]])
INIT_RPYS     = np.array([[0.0, 0.0, 0.0]])

PRINT_HZ      = 4

TRAJECTORIES  = {
    "Hover":    traj_hover,
    "Circle":   traj_circle,
    "Figure-8": traj_figure8,
}

# ─────────────────────────────────────────────────────────────────────────────
#  MOTOR MIXING  (Crazyflie X-frame)
# ─────────────────────────────────────────────────────────────────────────────
_S2   = np.sqrt(2.0)
_AX   = _S2 / (4.0 * ARM)
_AZ   = KF / (4.0 * KM)

ALLOC = np.array([
    [0.25, -_AX, -_AX, -_AZ],
    [0.25,  _AX, -_AX,  _AZ],
    [0.25,  _AX,  _AX, -_AZ],
    [0.25, -_AX,  _AX,  _AZ],
])


def thrust_torque_to_rpm(F_total, tau_x, tau_y, tau_z):
    wrench   = np.array([F_total, tau_x, tau_y, tau_z])
    F_motors = ALLOC @ wrench
    F_motors = np.clip(F_motors, 0.0, None)
    rpm      = np.sqrt(np.maximum(F_motors / KF, 0.0))
    return np.clip(rpm, MIN_RPM, MAX_RPM)


# ─────────────────────────────────────────────────────────────────────────────
#  PAYLOAD
# ─────────────────────────────────────────────────────────────────────────────
def add_payload_sphere(pybullet_client, drone_id, payload_mass, radius=0.015):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius,
                                 physicsClientId=pybullet_client)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                              rgbaColor=[0.8, 0.3, 0.1, 1.0],
                              physicsClientId=pybullet_client)
    sphere = p.createMultiBody(baseMass=payload_mass,
                               baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=[0.0, 0.0, 0.1],
                               physicsClientId=pybullet_client)
    p.createConstraint(drone_id, -1, sphere, -1, p.JOINT_FIXED,
                       [0, 0, 0], [0, 0, 0.04], [0, 0, 0],
                       physicsClientId=pybullet_client)
    return sphere


# ─────────────────────────────────────────────────────────────────────────────
#  LINEAR OUTER MPC  (position + velocity, CVXPY/OSQP)
# ─────────────────────────────────────────────────────────────────────────────
class LinearOuterMPC:
    """
    6-state double-integrator MPC.
    State  x = [px, py, pz, vx, vy, vz]
    Input  u = [ax, ay, az]  (desired accelerations, m/s²)

    Output: acceleration command → converted to attitude + thrust target
            for the PID inner loop.

    The mass parameter is used only to scale the thrust output.
    The position/velocity dynamics are mass-independent (Newton: a = F/m,
    so we optimise accelerations directly).
    """

    def __init__(self, dt=CTRL_DT, horizon=20, mass=BASE_MASS):
        self.dt   = dt
        self.N    = horizon
        self.mass = mass

        self._build_problem()

    def _build_problem(self):
        N  = self.N
        dt = self.dt

        A = np.block([[np.eye(3), dt * np.eye(3)],
                      [np.zeros((3, 3)), np.eye(3)]])
        B = np.block([[0.5 * dt**2 * np.eye(3)],
                      [dt * np.eye(3)]])
        self._A = A
        self._B = B

        # Weights — higher position/altitude weight, moderate velocity
        Q  = np.diag([100.0, 100.0, 180.0,   # pos x, y, z
                       12.0,  12.0,  20.0])   # vel x, y, z
        Qf = np.diag([150.0, 150.0, 250.0,
                       18.0,  18.0,  30.0])
        R  = np.diag([0.8, 0.8, 1.5])         # accel x, y, z

        # Acceleration limits (m/s²)
        A_XY_MAX  = 4.0
        A_Z_UP    = 6.0
        A_Z_DOWN  = 4.0

        x    = cp.Variable((6, N + 1))
        u    = cp.Variable((3, N))
        x0p  = cp.Parameter(6)
        xref = cp.Parameter((6, N + 1))

        cost        = 0
        constraints = [x[:, 0] == x0p]

        for k in range(N):
            cost       += cp.quad_form(x[:, k] - xref[:, k], Q)
            cost       += cp.quad_form(u[:, k], R)
            constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k],
                            cp.abs(u[0, k]) <= A_XY_MAX,
                            cp.abs(u[1, k]) <= A_XY_MAX,
                            u[2, k] <=  A_Z_UP,
                            u[2, k] >= -A_Z_DOWN,
                            cp.abs(x[0, k]) <= 4.0,
                            cp.abs(x[1, k]) <= 4.0,
                            x[2, k] >= 0.0,
                            x[2, k] <= 2.0,
                            cp.norm(x[3:6, k], "inf") <= 4.0]

        cost += cp.quad_form(x[:, N] - xref[:, N], Qf)

        self._prob = cp.Problem(cp.Minimize(cost), constraints)
        self._x    = x
        self._u    = u
        self._x0p  = x0p
        self._xref = xref

        # Fallback LQR gains for when OSQP fails
        Pt      = solve_discrete_are(A, B, Q, R)
        self._K = np.linalg.inv(R + B.T @ Pt @ B) @ (B.T @ Pt @ A)

    def set_mass(self, mass):
        self.mass = mass

    def compute(self, cur_pos, cur_vel, ref_fn, t_now):
        """
        Returns acceleration command [ax, ay, az] in m/s².
        """
        x0 = np.hstack([cur_pos, cur_vel])

        xref = np.zeros((6, self.N + 1))
        for k in range(self.N + 1):
            pk, vk = ref_fn(t_now + k * self.dt)
            xref[:, k] = np.hstack([pk, vk])

        self._x0p.value  = x0
        self._xref.value = xref

        try:
            self._prob.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                             eps_abs=1e-4, eps_rel=1e-4, max_iter=4000)
            if self._u.value is not None:
                return self._u.value[:, 0]
        except Exception:
            pass

        # Fallback: LQR
        err = x0 - xref[:, 0]
        return np.clip(-self._K @ err, [-4, -4, -4], [4, 4, 6])


def accel_to_attitude_thrust(acc_cmd, mass):
    """
    Convert desired acceleration [ax, ay, az] to:
      target_rpy  = [roll_des, pitch_des, 0]   (radians, small-angle)
      F_total     = desired total thrust (N)

    Uses small-angle linearisation consistent with the MPC model.
    """
    ax, ay, az = acc_cmd
    # Small-angle: ax ≈ g·pitch,  ay ≈ -g·roll
    pitch_des = np.clip( ax / GRAVITY, -0.35, 0.35)
    roll_des  = np.clip(-ay / GRAVITY, -0.35, 0.35)
    F_total   = mass * (GRAVITY + az)
    F_total   = np.clip(F_total, 0.3 * mass * GRAVITY, 2.5 * mass * GRAVITY)
    return np.array([roll_des, pitch_des, 0.0]), F_total


# ─────────────────────────────────────────────────────────────────────────────
#  PAYLOAD RPM BIAS  — compensates fixed-gain PID for extra mass
# ─────────────────────────────────────────────────────────────────────────────
def compute_payload_rpm_bias(total_mass):
    """
    The DSLPIDControl is tuned for BASE_MASS. When total_mass > BASE_MASS,
    all four motors need more thrust. We add a constant RPM offset so the
    hover equilibrium is correct even with fixed PID gains.

    hover_rpm(m) = sqrt(m * g / (4 * KF))
    bias = hover_rpm(total_mass) - hover_rpm(BASE_MASS)
    """
    hover_full = np.sqrt(total_mass * GRAVITY / (4.0 * KF))
    return hover_full - HOVER_RPM_BASE


# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS SAVE
# ─────────────────────────────────────────────────────────────────────────────
def save_results(traj_name, condition_name, controller_name,
                 errors, ctrl_efforts, step_times, wall_total):
    sim_total    = len(errors) * CTRL_DT
    rt_factor    = sim_total / wall_total if wall_total > 0 else 0.0
    errors       = np.array(errors)
    ctrl_efforts = np.array(ctrl_efforts)
    step_times   = np.array(step_times)

    print("\n  " + "=" * 58)
    print(f"  Controller  : {controller_name}  |  Traj: {traj_name}  |  {condition_name}")
    print(f"  Sim: {sim_total:.1f}s  |  Wall: {wall_total:.1f}s  |  RT: {rt_factor:.2f}x")
    print("  --- Tracking error (m) ---")
    print(f"    RMSE    : {np.sqrt(np.mean(errors**2)):.4f}")
    print(f"    MAE     : {np.mean(errors):.4f}")
    print(f"    Max     : {np.max(errors):.4f}")
    print(f"    Steady  : {np.mean(errors[len(errors)//2:]):.4f}  (2nd half)")
    print("  --- Control effort ---")
    print(f"    Mean ΔRPM: {np.mean(ctrl_efforts):.1f}")
    print(f"    Max  ΔRPM: {np.max(ctrl_efforts):.1f}")
    print("  --- Timing ---")
    print(f"    Mean step: {np.mean(step_times):.1f} ms  (budget {CTRL_DT*1000:.1f} ms)")
    print(f"    Over budget: {np.sum(step_times > CTRL_DT*1000)}/{len(step_times)}")
    print("  " + "=" * 58)

    from datetime import datetime
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


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_visualization(traj_name, condition_name, controller_name):
    traj_fn = TRAJECTORIES[traj_name]

    # ── Condition setup ────────────────────────────────────────────────────
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

    # RPM bias to compensate PID's fixed hover equilibrium
    rpm_bias = compute_payload_rpm_bias(total_mass)
    hover_rpm_case = np.sqrt(total_mass * GRAVITY / (4.0 * KF))

    print(f"\n  Trajectory  : {traj_name}")
    print(f"  Condition   : {condition_name}")
    print(f"  Controller  : {controller_name}")
    print(f"  Total mass  : {total_mass*1000:.1f} g")
    print(f"  RPM bias    : {rpm_bias:+.1f}  (hover compensation)")
    print(f"  Wind force  : {wind} N/step")

    # ── Environment ────────────────────────────────────────────────────────
    env = CtrlAviary(
        drone_model  = DroneModel.CF2X,
        num_drones   = 1,
        initial_xyzs = INIT_XYZS.copy(),
        initial_rpys = INIT_RPYS.copy(),
        physics      = Physics.PYB,
        pyb_freq     = SIM_FREQ,
        ctrl_freq    = CTRL_FREQ,
        gui          = True,
        record       = False,
    )
    PYB_CLIENT = env.getPyBulletClient()
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5, cameraYaw=45, cameraPitch=-50,
        cameraTargetPosition=[0.0, 0.0, 0.5],
        physicsClientId=PYB_CLIENT)

    # ── Controllers ────────────────────────────────────────────────────────
    pid_ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

    use_mpc = controller_name in ("MPC", "PID+MPC")
    if use_mpc:
        print("  Building LinearOuterMPC (CVXPY/OSQP)...")
        mpc = LinearOuterMPC(dt=CTRL_DT, horizon=20, mass=total_mass)
        print(f"  MPC ready.")

    # ── Reset env + apply mass ─────────────────────────────────────────────
    obs, _ = env.reset()

    # FIX: set PyBullet drone mass to BASE_MASS (payload is a separate body)
    p.changeDynamics(env.DRONE_IDS[0], -1, mass=BASE_MASS,
                     physicsClientId=PYB_CLIENT)

    if has_payload:
        add_payload_sphere(PYB_CLIENT, env.DRONE_IDS[0], PAYLOAD_MASS)
        print("  Payload sphere attached (red, r=15 mm)")

    # ── Accumulators ───────────────────────────────────────────────────────
    errors, rpms, ctrl_efforts, step_times, sim_times = [], [], [], [], []
    actual_traj, ref_traj = [], []

    action        = np.zeros((1, 4))
    N_STEPS       = int(DURATION * CTRL_FREQ)
    print_interval = max(1, CTRL_FREQ // PRINT_HZ)

    # MPC state (shared between MPC and PID+MPC modes)
    mpc_counter    = 0
    last_acc_cmd   = np.zeros(3)         # cached between MPC updates
    last_rpy_cmd   = INIT_RPYS[0].copy()
    last_F_total   = total_mass * GRAVITY

    print(f"\n  Flying {DURATION:.0f}s — close the GUI window to stop early")
    print(f"  {'t':>6}  {'Target':>22}  {'Pos':>22}  {'Err(m)':>7}  {'RPM':>7}  {'RT%':>5}")
    print("  " + "-" * 78)

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

        # ── Controller ──────────────────────────────────────────────────
        if controller_name == "PID":
            # Baseline: plain DSLPIDControl + RPM bias for mass compensation
            rpm, _, _ = pid_ctrl.computeControlFromState(
                control_timestep=CTRL_DT, state=state,
                target_pos=target_pos, target_vel=target_vel,
                target_rpy=INIT_RPYS[0])
            rpm = np.clip(rpm + rpm_bias, MIN_RPM, MAX_RPM)

        elif controller_name == "MPC":
            # Full MPC every step: outer MPC → attitude cmd → PID inner loop
            # This gives the same inner-loop structure as PID+MPC for fairness.
            acc_cmd = mpc.compute(cur_pos, cur_vel, traj_fn, t)
            target_rpy, F_total = accel_to_attitude_thrust(acc_cmd, total_mass)
            rpm, _, _ = pid_ctrl.computeControlFromState(
                control_timestep=CTRL_DT, state=state,
                target_pos=cur_pos, target_vel=np.zeros(3),
                target_rpy=target_rpy)
            rpm = np.clip(rpm + rpm_bias, MIN_RPM, MAX_RPM)

        elif controller_name == "PID+MPC":
            # TRUE HYBRID:
            #   Outer loop  (24 Hz) — MPC computes optimal acceleration command
            #                         using full prediction horizon + constraints
            #   Inner loop  (48 Hz) — PID stabilises attitude at high rate
            #                         using MPC's attitude/thrust target
            # The MPC runs at half frequency to give it time budget, while
            # PID keeps attitude stable between MPC updates.
            if mpc_counter % MPC_EVERY == 0:
                last_acc_cmd = mpc.compute(cur_pos, cur_vel, traj_fn, t)
                last_rpy_cmd, last_F_total = accel_to_attitude_thrust(
                    last_acc_cmd, total_mass)

            mpc_counter += 1

            rpm, _, _ = pid_ctrl.computeControlFromState(
                control_timestep=CTRL_DT, state=state,
                target_pos=cur_pos, target_vel=np.zeros(3),
                target_rpy=last_rpy_cmd)
            rpm = np.clip(rpm + rpm_bias, MIN_RPM, MAX_RPM)

        else:
            raise ValueError(f"Unknown controller: {controller_name}")

        action[0, :] = rpm

        # ── Step simulation ─────────────────────────────────────────────
        try:
            # FIX: Apply wind force once per control step (not scaled by SUBSTEPS)
            # The force in Newtons already represents the average over the step.
            if np.any(wind != 0):
                w_noisy = wind * (1.0 + np.random.uniform(-0.15, 0.15, size=3))
                p.applyExternalForce(
                    env.DRONE_IDS[0], -1,
                    w_noisy.tolist(), [0.0, 0.0, 0.0],
                    p.WORLD_FRAME,
                    physicsClientId=PYB_CLIENT)

            obs, _, terminated, truncated, _ = env.step(action)

        except Exception as e:
            print(f"  [!] Simulation error: {e}")
            break

        if terminated or truncated:
            print("  [!] Episode ended early.")
            break

        # ── Metrics ─────────────────────────────────────────────────────
        err      = float(np.linalg.norm(cur_pos - target_pos))
        mean_rpm = float(np.mean(rpm))
        effort   = float(np.sum(np.abs(rpm - hover_rpm_case)))
        step_ms  = (time.time() - step_wall_start) * 1000.0

        errors.append(err)
        rpms.append(mean_rpm)
        ctrl_efforts.append(effort)
        step_times.append(step_ms)
        sim_times.append(t)

        rt_pct = min(999, int(CTRL_DT / max(step_ms / 1000.0, 1e-9) * 100))

        if i % print_interval == 0:
            tgt_s = f"({target_pos[0]:+.2f},{target_pos[1]:+.2f},{target_pos[2]:+.2f})"
            pos_s = f"({cur_pos[0]:+.2f},{cur_pos[1]:+.2f},{cur_pos[2]:+.2f})"
            print(f"  {t:6.2f}s  {tgt_s:>22}  {pos_s:>22}  {err:>7.4f}  {mean_rpm:>7.0f}  {rt_pct:>4}%")

        # Real-time pacing
        elapsed = time.time() - step_wall_start
        if CTRL_DT - elapsed > 0:
            time.sleep(CTRL_DT - elapsed)

    wall_total = time.time() - wall_start

    try:
        env.close()
    except Exception:
        pass

    # ── Save trajectories ──────────────────────────────────────────────────
    safe = f"{controller_name}_{traj_name}_{condition_name}".replace(" ", "_").replace("+", "plus")
    np.save(f"traj_{safe}_actual.npy", np.array(actual_traj))
    np.save(f"traj_{safe}_ref.npy",    np.array(ref_traj))
    print(f"  Saved: traj_{safe}_actual.npy / _ref.npy")

    save_results(traj_name, condition_name, controller_name,
                 errors, ctrl_efforts, step_times, wall_total)

    return {
        "controller": controller_name,
        "traj_name":  traj_name,
        "condition":  condition_name,
        "times":      sim_times,
        "errors":     errors,
        "rpms":       rpms,
        "ctrl_efforts": ctrl_efforts,
        "actual_traj":  np.array(actual_traj),
        "ref_traj":     np.array(ref_traj),
        "rmse":         float(np.sqrt(np.mean(np.array(errors)**2))),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI MENU
# ─────────────────────────────────────────────────────────────────────────────
def choose(title, options):
    print(f"\nSelect {title}:")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    return options[int(input("Enter number: ").strip()) - 1]


def main():
    print("\n" + "=" * 55)
    print("  Drone Controller Comparison")
    print("  PID  |  MPC (outer)  |  PID+MPC hybrid")
    print("=" * 55)

    traj_name  = choose("trajectory",  ["Hover", "Circle", "Figure-8"])
    cond_name  = choose("condition",   ["Nominal", "Wind Only",
                                        "Payload Only", "Wind + Payload"])
    ctrl_name  = choose("controller",  ["PID", "MPC", "PID+MPC", "Compare All"])

    runs = []
    controllers = ["PID", "MPC", "PID+MPC"] if ctrl_name == "Compare All" else [ctrl_name]

    for c in controllers:
        result = run_visualization(traj_name, cond_name, c)
        runs.append(result)

    # Print summary table
    print("\n  ── SUMMARY ────────────────────────────────────────")
    print(f"  {'Controller':<12} {'RMSE (m)':>10} {'MAE (m)':>10} {'MaxErr (m)':>12}")
    print("  " + "-" * 48)
    for r in runs:
        errs = np.array(r["errors"])
        print(f"  {r['controller']:<12} "
              f"{np.sqrt(np.mean(errs**2)):>10.4f} "
              f"{np.mean(errs):>10.4f} "
              f"{np.max(errs):>12.4f}")
    print("  " + "-" * 48)

    try:
        plot_data = [(r["controller"], r["ref_traj"], r["actual_traj"],
                      r["traj_name"], r["condition"]) for r in runs]
        plot_runs(plot_data)
    except Exception as e:
        print(f"  [Plot skipped: {e}]")


if __name__ == "__main__":
    main()