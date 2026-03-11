import casadi as ca
import numpy as np

class QuadrotorMPC:
    def __init__(self):
        self.dt = 0.05
        self.N=10
        
        self.opti=ca.Opti()
        self.X=self.opti.variable(6,self.N+1)
        self.U=self.opti.variable(3,self.N)
        
        self.P_start = self.opti.parameter(6)     
        self.P_goal = self.opti.parameter(3)  
        g = 9.81
            
        for k in range(self.N): 
            self.opti.subject_to(self.X[0:3, k+1] == self.X[0:3, k] + self.X[3:6, k] * self.dt)
            self.opti.subject_to(self.X[3, k+1] == self.X[3, k] + g * self.U[1, k] * self.dt)
            self.opti.subject_to(self.X[4, k+1] == self.X[4, k] - g * self.U[0, k] * self.dt)
            self.opti.subject_to(self.X[5, k+1] == self.X[5, k] + (self.U[2, k] - g) * self.dt)
            
        self.opti.subject_to(self.opti.bounded(-0.35, self.U[0:2, :], 0.35))
        self.opti.subject_to(self.opti.bounded(0, self.U[2, :], 20))
        
        obj = ca.sumsqr(self.X[0:3, -1] - self.P_goal) + 0.1 * ca.sumsqr(self.U)
        self.opti.minimize(obj)
        
    def solve(self,current_state,target_pos):
        self.opti.set_value(self.P_start, current_state)
        self.opti.set_value(self.P_goal, target_pos)
        
        try:
            sol=self.opti.solve()
            return sol.value(self.U)[:, 0]
        except:
            return np.array([0.0, 0.0, 9.81])
        