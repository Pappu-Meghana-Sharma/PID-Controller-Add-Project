from drone_physics import DroneEnvironment
from control_logic import ResearchController
from mpc_planner import QuadrotorMPC
import numpy as np
import csv

def run_research_flight(base_mass, payload_mass, wind_force, target_pos=[1, 1, 1],
                        masses_grid=None, winds_grid=None, gains_grid=None):
    """
    Run one flight experiment.
    - base_mass: mass of drone without payload
    - payload_mass: added mass
    - wind_force: 3D wind force vector (world frame)
    - target_pos: desired position [x,y,z]
    - masses_grid, winds_grid, gains_grid: for 2D gain scheduler (optional)
    """
    env = DroneEnvironment(render=True)
    # Initialize controller with optional gain scheduler table
    if masses_grid is not None and winds_grid is not None and gains_grid is not None:
        ctrl = ResearchController(masses_grid, winds_grid, gains_grid)
    else:
        ctrl = ResearchController()  # fallback to 1D mass-only table
    ctrl.reset_state()
    mpc = QuadrotorMPC()

    drone = env.reset_drone(mass=base_mass)
    env.set_payload_mass(base_mass + payload_mass)  # simple total mass adjustment
    total_mass = env.get_mass()
    wind_magnitude = np.linalg.norm(wind_force)
    gains = ctrl.get_gains(total_mass, wind_magnitude)

    data_log = []   # will store detailed state at each step

    for i in range(1200):   # 5 seconds at 240 Hz
        # Get current state
        angles, rates = env.get_imu_data()
        pos = env.get_position()
        vel = env.get_linear_velocity()

        # MPC update (every 5 steps)
        if i % 5 == 0:
            current_state = np.concatenate([pos, vel])
            mpc_out = mpc.solve(current_state, target_pos)
            target_angles = mpc_out[0:2]   # roll, pitch (yaw target = 0)
            thrust_accel = mpc_out[2]       # vertical acceleration command

        # ML residual placeholder (to be filled later)
        torque_residual = np.zeros(3)

        # Cascaded PID to get torques
        dt = env.dt
        base_torque = ctrl.run_cascaded_control(
            np.array([target_angles[0], target_angles[1], 0.0]),  # yaw target = 0
            angles,
            rates,
            gains,
            dt
        )
        final_torque = base_torque + torque_residual

        # Convert thrust acceleration to force
        thrust_force = total_mass * thrust_accel

        # Apply wind and control
        env.wind_disturbances(force=wind_force)
        env.apply_control(thrust_force, final_torque)
        env.step()

        # Log data for trajectory analysis
        data_log.append({
            'time': i * dt,
            'pos_x': pos[0], 'pos_y': pos[1], 'pos_z': pos[2],
            'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2],
            'roll': angles[0], 'pitch': angles[1], 'yaw': angles[2],
            'rate_roll': rates[0], 'rate_pitch': rates[1], 'rate_yaw': rates[2],
            'target_pos_x': target_pos[0], 'target_pos_y': target_pos[1], 'target_pos_z': target_pos[2],
            'target_roll': target_angles[0], 'target_pitch': target_angles[1], 'target_yaw': 0.0,
            'thrust_force': thrust_force,
            'torque_x': final_torque[0], 'torque_y': final_torque[1], 'torque_z': final_torque[2],
            'wind_force_x': wind_force[0], 'wind_force_y': wind_force[1], 'wind_force_z': wind_force[2]
        })

    # Save data to CSV
    filename = f"exp_m{base_mass}_p{payload_mass}_w{wind_force[0]}.csv"
    save_to_csv(filename, data_log)
    env.close()
    return data_log

def save_to_csv(filename, data):
    if not data:
        return
    keys = data[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

# Example usage: define a 2D gain table (mass, wind) and run experiments
if __name__ == "__main__":
    try:
        data = np.load("gain_table.npz")
        masses_grid = data['masses'].tolist()
        winds_grid = data['wind_magnitudes'].tolist()
        gains_grid = data['gains_grid']          # shape (M, W, 6)
        print("Loaded gain table from gain_table.npz")
    except FileNotFoundError:
        print("No gain table found; falling back to mass‑only lookup.")
        masses_grid = winds_grid = gains_grid = None
        
        base_masses = [0.027, 0.045, 0.060]          # kg (Crazyflie base + variations)
        payloads = [0.0, 0.005, 0.010]               # kg (0, 5g, 10g – within max payload)
        
        wind_magnitude = 0.2                          # N (strong gust)
        wind_directions = [[1,0,0], [0,1,0], [-1,0,0], [0,-1,0]]  # +x, +y, -x, -y

        # Loop over all combinations
        for bm in base_masses:
            for p in payloads:
                for direction in wind_directions:
                    wf = [wind_magnitude * d for d in direction]
                    run_research_flight(bm, p, wf,
                                        masses_grid=masses_grid,
                                        winds_grid=winds_grid,
                                        gains_grid=gains_grid)

    