from drone_physics import DroneEnvironment
from control_logic import ResearchController
import csv
import pybullet as p
import numpy as np
from mpc_planner import QuadrotorMPC

def run_research_flight(mass, wind_force, target_pos=[1, 1, 1]):
    env = DroneEnvironment(render=True)
    ctrl = ResearchController()
    mpc = QuadrotorMPC()
    drone = env.reset_drone(mass=mass)
    gains = ctrl.get_gains(mass)
    
    data_log = []
    
    for i in range(1200):
        current_angles, current_rates = env.get_state()
        current_pos = env.get_position()
        if i%5==0:
            target_angles=mpc.solve(current_pos,target_pos)
        torque_residual=np.zeros(3)
        base_torque=ctrl.run_cascaded_control(
            target_angles, 
            current_angles, 
            current_rates, 
            gains
        )
        
        final_torque = base_torque + torque_residual
        
        env.apply_disturbances(force=wind_force)
        env.apply_control(final_torque)
        env.step()
        
        data_log.append([i/240.0, target_pos, current_pos, target_angles, current_angles])
    save_to_csv(f"exp_m{mass}_w{wind_force[0]}.csv", data_log)
    env.close()

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Target_Pos", "Actual_Pos", "Target_Angle", "Actual_Angle"])
        writer.writerows(data)