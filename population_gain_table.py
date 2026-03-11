import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import os
import json


from drone_physics import DroneEnvironment
from control_logic import ResearchController
from mpc_planner import QuadrotorMPC


PAYLOAD = 0.0     
TARGET_POS = [1.0, 1.0, 1.0]
SIM_STEPS = 1200            
DT = 1/240.0

W_POS = 1.0
W_ATT = 0.1
W_CTRL = 0.01
CRASH_PENALTY = 1e6

N_CALLS = 50                  # total evaluations per grid point (adjust as needed)
N_INITIAL = 10                 # random initial points
RANDOM_STATE = 42              # for reproducibility

# Search space bounds (same for all grid points)
space = [
    Real(5.0, 25.0, name='kp_a'),
    Real(0.0, 1.0, name='ki_a'),
    Real(0.0, 0.5, name='kd_a'),
    Real(0.1, 2.0, name='kp_r'),
    Real(0.0, 0.5, name='ki_r'),
    Real(0.0, 0.2, name='kd_r')
]

def simulate_with_gains(gains, mass, wind_force, render=False):
    """
    Run one simulation with given gains and return a cost.
    gains: list of 6 floats [Kp_a, Ki_a, Kd_a, Kp_r, Ki_r, Kd_r]
    wind_force: 3D wind force vector (world frame)
    """
    env = DroneEnvironment(render=render)
    drone = env.reset_drone(mass=mass)
    env.set_payload_mass(mass)          # no extra payload for tuning
    total_mass = env.get_mass()

    ctrl = ResearchController()          # will use gains directly
    ctrl.reset_state()
    mpc = QuadrotorMPC()

    pos_errors = []
    att_errors = []
    control_effort = []

    for i in range(SIM_STEPS):
        angles, rates = env.get_imu_data()
        pos = env.get_position()
        vel = env.get_linear_velocity()

        if i % 5 == 0:
            state = np.concatenate([pos, vel])
            mpc_out = mpc.solve(state, TARGET_POS)
            target_angles = mpc_out[0:2]
            thrust_accel = mpc_out[2]

        dt = env.dt
        torque = ctrl.run_cascaded_control(
            np.array([target_angles[0], target_angles[1], 0.0]),
            angles,
            rates,
            gains,
            dt
        )

        thrust_force = total_mass * thrust_accel

        env.wind_disturbances(force=wind_force)
        env.apply_control(thrust_force, torque)
        env.step()

        pos_error = np.linalg.norm(pos - TARGET_POS)
        att_error = np.linalg.norm(angles - np.array([target_angles[0], target_angles[1], 0.0]))
        ctrl_eff = np.sum(np.square(torque))

        pos_errors.append(pos_error)
        att_errors.append(att_error)
        control_effort.append(ctrl_eff)

        if pos[2] < 0.0 or np.linalg.norm(pos) > 10.0:
            env.close()
            return CRASH_PENALTY

    env.close()

    cost = (W_POS * np.sqrt(np.mean(np.square(pos_errors))) +
            W_ATT * np.sqrt(np.mean(np.square(att_errors))) +
            W_CTRL * np.mean(control_effort))
    return cost


def create_objective(mass, wind_force):
    """Create an objective function for skopt with fixed mass and wind."""
    @use_named_args(space)
    def objective(**params):
        gains = [params['kp_a'], params['ki_a'], params['kd_a'],
                 params['kp_r'], params['ki_r'], params['kd_r']]
        cost = simulate_with_gains(gains, mass, wind_force, render=False)
        return cost
    return objective


def tune_for_condition(mass, wind_magnitude, wind_direction):
    """
    Run Bayesian optimization for one (mass, wind) condition.
    wind_direction is a unit vector; we scale by wind_magnitude.
    Returns the best gains (list of 6 floats).
    """
    wind_force = (np.array(wind_direction) * wind_magnitude).tolist()
    print(f"\n--- Tuning for mass={mass:.3f} kg, wind={wind_force} ---")

    objective = create_objective(mass, wind_force)

    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL,
        acq_func='EI',
        random_state=RANDOM_STATE,
        verbose=False
    )

    best_gains = result.x
    best_cost = result.fun
    print(f"Best gains: {[round(g,4) for g in best_gains]}, cost={best_cost:.3f}")
    return best_gains


if __name__ == "__main__":
   
    masses = [0.027, 0.045, 0.060]         
    wind_magnitudes = [0.0, 0.1, 0.2]        
    # For simplicity, we assume wind always blows in +x direction.
    # If you want multiple directions, add another loop over directions.
    wind_direction = [1.0, 0.0, 0.0]

    # Create a 3D array to store gains: shape (len(masses), len(wind_mags), 6)
    gains_grid = np.zeros((len(masses), len(wind_magnitudes), 6))

    for i, m in enumerate(masses):
        for j, wmag in enumerate(wind_magnitudes):
            gains = tune_for_condition(m, wmag, wind_direction)
            gains_grid[i, j, :] = gains

    # Save the table and metadata
    np.savez("gain_table.npz",
             masses=masses,
             wind_magnitudes=wind_magnitudes,
             gains_grid=gains_grid,
             wind_direction=wind_direction)

    # Also save as JSON for human readability (optional)
    data = {
        "masses": masses,
        "wind_magnitudes": wind_magnitudes,
        "wind_direction": wind_direction,
        "gains_grid": gains_grid.tolist()
    }
    with open("gain_table.json", "w") as f:
        json.dump(data, f, indent=2)

    print("\nGain table saved to gain_table.npz and gain_table.json")