# mpc_planner.py
import casadi as ca
import numpy as np

class QuadrotorMPC:
    """
    Model Predictive Controller that outputs desired roll, pitch and thrust
    acceleration to reach a target position.
    Uses a simplified linearized model.
    """
    def __init__(self):
        self.dt = 0.05          # MPC time step (50 ms)
        self.N = 10              # prediction horizon

        self.opti = ca.Opti()
        # State: [x, y, z, vx, vy, vz]
        self.X = self.opti.variable(6, self.N + 1)
        # Control: [roll, pitch, thrust_accel] (thrust_accel in m/s²)
        self.U = self.opti.variable(3, self.N)

        self.P_start = self.opti.parameter(6)
        self.P_goal = self.opti.parameter(3)
        g = 9.81

        # Initial condition
        self.opti.subject_to(self.X[:, 0] == self.P_start)

        # Dynamics (simplified, linearized around hover)
        for k in range(self.N):
            self.opti.subject_to(self.X[0:3, k+1] == self.X[0:3, k] + self.X[3:6, k] * self.dt)
            self.opti.subject_to(self.X[3, k+1] == self.X[3, k] + g * self.U[1, k] * self.dt)
            self.opti.subject_to(self.X[4, k+1] == self.X[4, k] - g * self.U[0, k] * self.dt)
            self.opti.subject_to(self.X[5, k+1] == self.X[5, k] + (self.U[2, k] - g) * self.dt)

        # Control limits (roll/pitch ±0.35 rad ≈ 20°, thrust between 0 and 20 m/s²)
        self.opti.subject_to(self.opti.bounded(-0.35, self.U[0:2, :], 0.35))
        self.opti.subject_to(self.opti.bounded(0, self.U[2, :], 20))

        # Objective: terminal position error + control effort
        obj = ca.sumsqr(self.X[0:3, -1] - self.P_goal) + 0.1 * ca.sumsqr(self.U)
        self.opti.minimize(obj)

        # Solver options (IPOPT, silent)
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver('ipopt', opts)

    def solve(self, current_state, target_pos):
        """
        current_state: 6‑dim [x,y,z,vx,vy,vz]
        target_pos: 3‑dim [x,y,z]
        returns: first control [roll, pitch, thrust_accel] as numpy array
        """
        self.opti.set_value(self.P_start, current_state)
        self.opti.set_value(self.P_goal, target_pos)

        try:
            sol = self.opti.solve()
            return sol.value(self.U)[:, 0]
        except Exception as e:
            print(f"MPC failed: {e}")
            return np.array([0.0, 0.0, 9.81]) 