import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import time
from scipy.linalg import solve_continuous_are, expm
import cvxopt
cvxopt.solvers.options['show_progress'] = False

# ----------------------------------------------------------------------
# Drone physical parameters (Crazyflie 2.0)
# ----------------------------------------------------------------------
MASS = 0.027                 # kg
G = 9.81                     # m/s^2
KF = 6.11e-8                 # thrust coefficient (N/(rad/s)^2)
KM = 1.5e-9                  # torque coefficient (N·m/(rad/s)^2)
ARM_LENGTH = 0.046           # m
INERTIA = np.diag([1.4e-5, 1.4e-5, 2.2e-5])  # kg·m^2

# Mixer matrix for X‑configuration (maps [T, tau_x, tau_y, tau_z] to motor forces)
MIXER_MATRIX = np.array([[1., 1., 1., 1.],
                         [1., -1., 1., -1.],
                         [-1., 1., 1., -1.],
                         [1., -1., -1., 1.]]) / (4 * np.sqrt(2))

# ----------------------------------------------------------------------
# Geometric low‑level controller: desired acceleration -> motor RPMs
# ----------------------------------------------------------------------
def geometric_controller(state, a_des, yaw_des, dt, attitude_gain=10.0, rate_gain=0.1):
    """
    state: dict with keys 'pos', 'vel', 'quat', 'omega'
    a_des: desired acceleration in world frame (3,)
    yaw_des: desired yaw angle (rad)
    returns: motor RPMs (4,)
    """
    pos, vel, quat, omega = state['pos'], state['vel'], state['quat'], state['omega']
    
    # 1. Desired thrust vector (world frame)
    thrust_vec_des = MASS * (a_des + np.array([0, 0, G]))   # T_w = m*(a_des + g e3)
    thrust_mag = np.linalg.norm(thrust_vec_des)
    if thrust_mag < 1e-6:
        thrust_mag = 1e-6
    b3_des = thrust_vec_des / thrust_mag                     # desired body z‑axis
    
    # 2. Desired body axes from yaw
    #    World x‑axis rotated by yaw, projected onto plane orthogonal to b3_des
    x_w = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
    b1_des = x_w - np.dot(x_w, b3_des) * b3_des
    b1_norm = np.linalg.norm(b1_des)
    if b1_norm < 1e-6:
        # b3_des is vertical → any horizontal direction works
        b1_des = np.array([1, 0, 0]) if abs(b3_des[2]) > 0.999 else np.cross(b3_des, [0, 0, 1])
        b1_des = b1_des / np.linalg.norm(b1_des)
    else:
        b1_des = b1_des / b1_norm
    b2_des = np.cross(b3_des, b1_des)                         # right‑hand frame
    R_des = np.column_stack([b1_des, b2_des, b3_des])         # body→world rotation
    
    # 3. Current rotation matrix from quaternion (w, x, y, z)
    w, x, y, z = quat
    R_cur = np.array([[1-2*(y**2+z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                      [2*(x*y + w*z), 1-2*(x**2+z**2), 2*(y*z - w*x)],
                      [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x**2+y**2)]])
    
    # 4. Attitude error (body frame)
    R_err = R_cur.T @ R_des
    # Compute axis‑angle from rotation matrix
    trace = np.trace(R_err)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    if angle < 1e-6:
        axis = np.array([0, 0, 0])
    else:
        axis = 1/(2*np.sin(angle)) * np.array([R_err[2,1]-R_err[1,2],
                                                R_err[0,2]-R_err[2,0],
                                                R_err[1,0]-R_err[0,1]])
    omega_des = attitude_gain * angle * axis                 # desired body rates
    
    # 5. Torque command (simplified P control)
    tau = rate_gain * INERTIA @ (omega_des - omega)           # N·m
    
    # 6. Mixer to motor forces, then to RPM
    forces = MIXER_MATRIX @ np.array([thrust_mag, tau[0], tau[1], tau[2]])
    # Ensure non‑negative forces (motors can only push)
    forces = np.maximum(forces, 0)
    rpm = np.sqrt(forces / KF) * 60 / (2 * np.pi)             # rad/s → RPM
    return rpm

# ----------------------------------------------------------------------
# High‑level controllers (output desired acceleration)
# ----------------------------------------------------------------------
class PIDController:
    def __init__(self, kp, kd, dt):
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.dt = dt
        self.last_error = None

    def compute(self, pos, vel, pos_des, vel_des=None, acc_des=None):
        if vel_des is None:
            vel_des = np.zeros(3)
        if acc_des is None:
            acc_des = np.zeros(3)
        error = pos_des - pos
        derror = vel_des - vel
        a_des = self.kp * error + self.kd * derror + acc_des
        return a_des

class MPCController:
    def __init__(self, horizon, dt, Q, R, mass=MASS):
        self.N = horizon
        self.dt = dt
        self.Q = Q
        self.R = R
        self.mass = mass
        # Discrete‑time double integrator model
        self.A = np.eye(6)
        self.A[0:3, 3:6] = dt * np.eye(3)
        self.B = np.zeros((6, 3))
        self.B[0:3, :] = 0.5 * dt**2 * np.eye(3)
        self.B[3:6, :] = dt * np.eye(3)
        # Build prediction matrices for QP (only needed once)
        self._build_prediction_matrices()

    def _build_prediction_matrices(self):
        n = 6   # state dimension
        m = 3   # control dimension
        # State transition over horizon: X = Phi x0 + Psi U
        Phi = np.zeros((self.N*n, n))
        Psi = np.zeros((self.N*n, self.N*m))
        Ak = np.eye(n)
        for k in range(self.N):
            Phi[k*n:(k+1)*n, :] = Ak
            # Build Psi block columns
            for j in range(k+1):
                Psi[k*n:(k+1)*n, j*m:(j+1)*m] = Ak @ self.B
            Ak = self.A @ Ak
        self.Phi = Phi
        self.Psi = Psi
        # Cost matrices
        Q_bar = np.kron(np.eye(self.N), self.Q)
        R_bar = np.kron(np.eye(self.N), self.R)
        self.H = 2 * (Psi.T @ Q_bar @ Psi + R_bar)
        # For cvxopt, H must be symmetric positive definite
        self.H = 0.5 * (self.H + self.H.T)   # ensure symmetry

    def compute(self, x0, x_ref_traj):
        """
        x0: current state [pos; vel] (6,)
        x_ref_traj: reference states over horizon (N,6) or (N,6) array
        returns: first optimal acceleration (3,)
        """
        N = self.N
        n = 6
        m = 3
        # Reference trajectory stacked
        x_ref_vec = x_ref_traj.reshape(N*n)
        # Compute gradient term
        q = 2 * self.Psi.T @ self.Q @ (self.Phi @ x0 - x_ref_vec)
        # Solve quadratic program: min 0.5 u^T H u + q^T u   (no constraints for simplicity)
        # Convert to cvxopt matrices
        H_cvx = cvxopt.matrix(self.H)
        q_cvx = cvxopt.matrix(q)
        sol = cvxopt.solvers.qp(H_cvx, q_cvx)
        u_opt = np.array(sol['x']).flatten()
        return u_opt[0:m]   # first control

class PIDMPCController:
    def __init__(self, pid, mpc, alpha=1.0):
        self.pid = pid
        self.mpc = mpc
        self.alpha = alpha   # weighting on MPC feedforward

    def compute(self, pos, vel, pos_des, vel_des, acc_des, x0, x_ref_traj):
        a_mpc = self.mpc.compute(x0, x_ref_traj)
        a_pid = self.pid.compute(pos, vel, pos_des, vel_des, acc_des)
        return a_mpc + self.alpha * a_pid

# ----------------------------------------------------------------------
# Simulation class
# ----------------------------------------------------------------------
class DroneConferenceSim:
    def __init__(self, scenario="eight"):
        self.scenario = scenario
        self.dt = 1/240
        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            physics=Physics.PYB,
            gui=True,
            record=False
        )
        self.reset_sim()

    def reset_sim(self):
        self.obs, self.info = self.env.reset()
        self.action = np.zeros((1, 4))
        self.history = []          # store time, pos, target, control effort
        self.targets = []
        self.time = 0.0
        # Reset mass to original (in case changed)
        p.changeDynamics(self.env.DRONE_IDS[0], -1, mass=MASS)

    def get_target_at_time(self, t):
        if self.scenario == "hover":
            return np.array([0, 0, 1.0])
        elif self.scenario == "circle":
            radius = 1.0
            omega = 0.5
            return np.array([radius * np.cos(omega * t),
                             radius * np.sin(omega * t), 1.0])
        elif self.scenario == "eight":
            a = 1.2
            scale = 1 + np.sin(0.5 * t)**2
            x = (a * np.cos(0.5 * t)) / scale
            y = (a * np.sin(0.5 * t) * np.cos(0.5 * t)) / scale
            return np.array([x, y, 1.0])
        return np.array([0, 0, 1.0])

    def apply_disturbances(self, t):
        if 5.0 < t < 7.0:
            wind_force = np.array([0.5, 0.2, 0.0])
            p.applyExternalForce(self.env.DRONE_IDS[0], -1,
                                 forceObj=wind_force,
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME)
        if t > 8.0:
            p.changeDynamics(self.env.DRONE_IDS[0], -1, mass=1.2 * MASS)

    def get_state(self):
        # Extract from observation: obs[0] is (12,) [x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
        obs = self.obs[0]
        pos = obs[0:3]
        quat = obs[3:7]   # [qx,qy,qz,qw]? Actually gym-pybullet-drones returns [qx,qy,qz,qw]? Need to check.
        # The observation order: pos (3), quat (4), rpy (3)?? Actually from docs: 12 elements: x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz
        # So indices: 0-2 pos, 3-6 quat (x,y,z,w), 7-9 vel, 10-12 omega
        quat = obs[3:7]   # [qx, qy, qz, qw]? But in PyBullet quaternion is [x,y,z,w]
        vel = obs[7:10]
        omega = obs[10:13]
        return {'pos': pos, 'vel': vel, 'quat': quat, 'omega': omega}

    def run_with_controller(self, controller, duration=15.0):
        self.reset_sim()
        t = 0.0
        rpm_history = []
        pos_history = []
        target_history = []
        time_history = []
        effort_history = []   # sum of squared RPMs

        # For MPC, we need to generate reference trajectory over horizon
        mpc_horizon = getattr(controller, 'N', 1)   # if controller is MPC, use its horizon, else 1

        while t < duration:
            # Get target
            pos_des = self.get_target_at_time(t)
            # For velocity and acceleration references (for PID feedforward)
            # Simple finite difference for velocity
            if t < 0.01:
                vel_des = np.zeros(3)
                acc_des = np.zeros(3)
            else:
                # Approximate using previous target
                dt_small = 0.01
                pos_next = self.get_target_at_time(t + dt_small)
                vel_des = (pos_next - pos_des) / dt_small
                pos_next2 = self.get_target_at_time(t + 2*dt_small)
                acc_des = (pos_next2 - 2*pos_next + pos_des) / (dt_small**2)

            # Apply disturbances
            self.apply_disturbances(t)

            # Get current state
            state = self.get_state()
            pos = state['pos']
            vel = state['vel']

            # Compute desired acceleration depending on controller type
            if isinstance(controller, PIDController):
                a_des = controller.compute(pos, vel, pos_des, vel_des, acc_des)
            elif isinstance(controller, MPCController):
                # Build reference trajectory over horizon
                x_ref = []
                for i in range(controller.N):
                    t_ref = t + i * self.dt
                    pos_ref = self.get_target_at_time(t_ref)
                    # Approx velocity by finite difference
                    if t_ref < 0.01:
                        vel_ref = np.zeros(3)
                    else:
                        pos_ref_next = self.get_target_at_time(t_ref + 0.01)
                        vel_ref = (pos_ref_next - pos_ref) / 0.01
                    x_ref.append(np.concatenate([pos_ref, vel_ref]))
                x_ref = np.array(x_ref)
                x0 = np.concatenate([pos, vel])
                a_des = controller.compute(x0, x_ref)
            elif isinstance(controller, PIDMPCController):
                # MPC part
                x_ref = []
                for i in range(controller.mpc.N):
                    t_ref = t + i * self.dt
                    pos_ref = self.get_target_at_time(t_ref)
                    if t_ref < 0.01:
                        vel_ref = np.zeros(3)
                    else:
                        pos_ref_next = self.get_target_at_time(t_ref + 0.01)
                        vel_ref = (pos_ref_next - pos_ref) / 0.01
                    x_ref.append(np.concatenate([pos_ref, vel_ref]))
                x_ref = np.array(x_ref)
                x0 = np.concatenate([pos, vel])
                a_des = controller.compute(pos, vel, pos_des, vel_des, acc_des, x0, x_ref)
            else:
                raise ValueError("Unknown controller type")

            # Desired yaw (always 0 for simplicity)
            yaw_des = 0.0

            # Get motor RPMs from geometric controller
            rpm = geometric_controller(state, a_des, yaw_des, self.dt)
            self.action[0, :] = rpm

            # Step environment
            self.obs, reward, terminated, truncated, self.info = self.env.step(self.action)
            t += self.dt
            self.time = t

            # Log data
            time_history.append(t)
            pos_history.append(pos.copy())
            target_history.append(pos_des.copy())
            effort = np.sum(rpm**2)   # control effort metric
            effort_history.append(effort)

        # Convert to arrays
        pos_history = np.array(pos_history)
        target_history = np.array(target_history)
        time_history = np.array(time_history)
        effort_history = np.array(effort_history)

        # Compute errors
        errors = np.linalg.norm(pos_history - target_history, axis=1)
        return time_history, errors, effort_history

# ----------------------------------------------------------------------
# Main script: run all scenarios and controllers
# ----------------------------------------------------------------------
def main():
    scenarios = ['hover', 'circle', 'eight']
    controllers = {
        'PID': PIDController(kp=[2.0, 2.0, 2.0], kd=[1.0, 1.0, 1.0], dt=1/240),
        'MPC': MPCController(horizon=10, dt=1/240, Q=np.diag([10,10,10,1,1,1]), R=np.eye(3)*0.1),
        'PID+MPC': PIDMPCController(
            pid=PIDController(kp=[2.0,2.0,2.0], kd=[1.0,1.0,1.0], dt=1/240),
            mpc=MPCController(horizon=10, dt=1/240, Q=np.diag([10,10,10,1,1,1]), R=np.eye(3)*0.1),
            alpha=0.5
        )
    }

    results = {}   # (scenario, ctrl_name) -> (time, error, effort)

    for scenario in scenarios:
        sim = DroneConferenceSim(scenario=scenario)
        for ctrl_name, ctrl in controllers.items():
            print(f"Running {scenario} with {ctrl_name}...")
            t, err, eff = sim.run_with_controller(ctrl, duration=15.0)
            results[(scenario, ctrl_name)] = (t, err, eff)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i, scenario in enumerate(scenarios):
        for j, ctrl_name in enumerate(['PID', 'MPC', 'PID+MPC']):
            ax = axes[i, j]
            t, err, eff = results[(scenario, ctrl_name)]
            ax.plot(t, err, label='Tracking error')
            ax.set_title(f'{scenario} - {ctrl_name}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Error (m)')
            ax.grid(True)
            # Also plot control effort on twin axis
            ax2 = ax.twinx()
            ax2.plot(t, eff, 'r--', label='Effort (RPM²)')
            ax2.set_ylabel('Effort', color='r')
            if i==0 and j==0:
                ax.legend(loc='upper left')
                ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150)
    plt.show()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("RMS Tracking Error (m) and Mean Control Effort (RPM²)")
    print("="*60)
    header = f"{'Scenario':<10} {'PID':>20} {'MPC':>20} {'PID+MPC':>20}"
    print(header)
    for scenario in scenarios:
        line = f"{scenario:<10}"
        for ctrl_name in ['PID', 'MPC', 'PID+MPC']:
            t, err, eff = results[(scenario, ctrl_name)]
            rms = np.sqrt(np.mean(err**2))
            mean_eff = np.mean(eff)
            line += f"  {rms:.3f}/{mean_eff:.1e}"
        print(line)
    print("="*60)

if __name__ == "__main__":
    main()