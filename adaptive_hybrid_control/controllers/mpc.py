import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are
from scipy.spatial.transform import Rotation
from ..config import CTRL_DT, BASE_MASS, GRAVITY

class LinearOuterMPC:
    """
    6-state double-integrator MPC.
    State  x = [px, py, pz, vx, vy, vz]
    Input  u = [ax, ay, az]  (desired accelerations, m/s²)
    """
    def __init__(self, dt=CTRL_DT, horizon=20, mass=BASE_MASS):
        self.dt    = dt
        self.N     = horizon
        self.mass  = mass
        self.d_est = np.zeros(3)
        self._P    = None    # Terminal cost matrix (returned to LyapunovMonitor)
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

        Q  = np.diag([100.0, 100.0, 200.0,
                       25.0,  25.0,  40.0])
        Qf = np.diag([150.0, 150.0, 300.0,
                       30.0,  30.0,  50.0])
        R  = np.diag([1.5, 1.5, 2.0])

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

        # LQR solution — also exported as Lyapunov P matrix
        Pt      = solve_discrete_are(A, B, Q, R)
        self._K = np.linalg.inv(R + B.T @ Pt @ B) @ (B.T @ Pt @ A)
        self._P = Pt    # ← used by LyapunovMonitor

    @property
    def P_matrix(self):
        """Positive-definite Lyapunov matrix from LQR DARE solution."""
        return self._P

    def set_mass(self, mass):
        self.mass = mass

    def compute(self, cur_pos, cur_vel, ref_fn, t_now):
        x0 = np.hstack([cur_pos, cur_vel])

        xref = np.zeros((6, self.N + 1))
        for k in range(self.N + 1):
            pk, vk = ref_fn(t_now + k * self.dt)
            xref[:, k] = np.hstack([pk, vk])

        self._x0p.value  = x0
        self._xref.value = xref

        acc_cmd = None
        try:
            self._prob.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                             eps_abs=1e-4, eps_rel=1e-4, max_iter=4000)
            if self._u.value is not None:
                acc_cmd = self._u.value[:, 0]
        except Exception as e:
            print(f"  [WARN] MPC fallback: {e}")

        if acc_cmd is None:
            err = x0 - xref[:, 0]
            acc_cmd = np.clip(-self._K @ err, [-4, -4, -4], [4, 4, 6])

        # Disturbance integrator
        err_p = xref[0:3, 0] - cur_pos
        self.d_est      = self.d_est + err_p * self.dt
        self.d_est[0:2] = np.clip(self.d_est[0:2], -2.0, 2.0)
        self.d_est[2]   = np.clip(self.d_est[2], -10.0, 10.0)

        # Disturbance acceleration addition
        K_i      = np.array([0.1, 0.1, 1.0])
        dist_acc = K_i * self.d_est
        dist_acc[0:2] = np.clip(dist_acc[0:2], -1.0, 1.0)
        dist_acc[2]   = np.clip(dist_acc[2], -2.0, 6.0)

        return acc_cmd + dist_acc


def accel_to_attitude_thrust(acc_cmd, mass, yaw=0.0):
    g_vec  = np.array([0.0, 0.0, GRAVITY])
    a_des  = acc_cmd + g_vec
    norm_a = np.linalg.norm(a_des)
    target_z_ax = a_des / norm_a if norm_a > 1e-6 else np.array([0.0, 0.0, 1.0])

    target_x_c     = np.array([np.cos(yaw), np.sin(yaw), 0])
    y_cross        = np.cross(target_z_ax, target_x_c)
    y_cross_norm   = np.linalg.norm(y_cross)
    target_y_ax    = (y_cross / y_cross_norm if y_cross_norm > 1e-6
                      else np.array([-np.sin(yaw), np.cos(yaw), 0.0]))
    target_x_ax    = np.cross(target_y_ax, target_z_ax)
    target_rotation = np.vstack([target_x_ax, target_y_ax, target_z_ax]).T
    target_euler    = Rotation.from_matrix(target_rotation).as_euler('XYZ', degrees=False)

    F_total = np.clip(mass * norm_a, 0.3 * mass * GRAVITY, 2.5 * mass * GRAVITY)
    return target_euler, F_total
