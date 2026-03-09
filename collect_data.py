import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
import logging

# ===========================
# CONFIGURATION
# ===========================
NUM_EPISODES = 100               # Increase for larger dataset
EPISODE_LENGTH_SEC = 20
INCLUDE_TARGET_IN_FEATURES = True
RANDOM_SEED = 42
MASS_MIN, MASS_MAX = 0.025, 0.056   # exploit the full safe range
WIND_MIN, WIND_MAX = -0.004, 0.004  # proven safe


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def get_randomized_target(t, episode_seed):
    rng = np.random.RandomState(episode_seed)
    r = rng.uniform(0.4, 0.8)
    w = rng.uniform(0.5, 1.0)
    step_x = rng.uniform(-0.5, 0.5)

    if t < 5:
        return np.array([0, 0, 1.0]), np.zeros(3)
    elif t < 12:
        return np.array([step_x, 0.3, 0.9]), np.zeros(3)
    else:
        t_adj = t - 12
        pos = np.array([r * np.cos(w * t_adj), r * np.sin(w * t_adj), 0.8])
        vel = np.array([-r * w * np.sin(w * t_adj), r * w * np.cos(w * t_adj), 0])
        return pos, vel

def run_collection():
    np.random.seed(RANDOM_SEED)
    X, Y = [], []
    crashed = []
    steps_per_ep = int(EPISODE_LENGTH_SEC * 48)

    for ep in range(NUM_EPISODES):
        mass = np.random.uniform(MASS_MIN, MASS_MAX)
        wx, wy = np.random.uniform(WIND_MIN, WIND_MAX, 2)

        env = CtrlAviary(DroneModel.CF2X, initial_xyzs=[[0,0,0.7]],
                         physics=Physics.PYB_GND_DRAG_DW, gui=False)
        client = env.getPyBulletClient()
        ctrl = DSLPIDControl(DroneModel.CF2X)
        p.changeDynamics(env.DRONE_IDS[0], -1, mass=mass, physicsClientId=client)

        obs, _ = env.reset()
        dt = env.CTRL_TIMESTEP

        for step in range(steps_per_ep):
            t = step * dt
            s = obs[0]
            # Ground truth
            true_pos, true_vel = s[0:3], s[10:13]
            true_rpy, true_angvel = s[7:10], s[13:16]

            # Noisy features
            noise = np.random.normal(0, 0.002, 12)
            state_feat = np.hstack([true_pos, true_vel, true_rpy, true_angvel]) + noise

            target_pos, target_vel = get_randomized_target(t, ep)

            action_raw, _, _ = ctrl.computeControlFromState(
                dt, obs[0], target_pos, target_vel)
            action = action_raw.flatten()

            p.applyExternalForce(env.DRONE_IDS[0], -1, [wx, wy, 0], [0,0,0],
                                 p.WORLD_FRAME, physicsClientId=client)

            obs, _, _, _, _ = env.step(action.reshape(1,4))
            s_next = obs[0]
            next_vel, next_angvel = s_next[10:13], s_next[13:16]
            next_pos = s_next[0:3]

            # Targets: linear & angular acceleration
            lin_acc = (next_vel - true_vel) / dt
            ang_acc = (next_angvel - true_angvel) / dt
            target = np.hstack([lin_acc, ang_acc])

            if INCLUDE_TARGET_IN_FEATURES:
                feat = np.hstack([state_feat, action, target_pos, target_vel, [mass, wx, wy]])
            else:
                feat = np.hstack([state_feat, action,[ mass, wx, wy]])

            X.append(feat)
            Y.append(target)

            if next_pos[2] < 0.05:
                crashed.append((ep, step))
                logging.warning(f"Episode {ep+1} crashed at step {step}")
                break

        env.close()
        logging.info(f"Episode {ep+1}/{NUM_EPISODES} done - mass={mass:.3f}")

        if (ep+1) % 10 == 0:
            np.savez(f"temp_ep{ep+1}.npz", X=np.array(X), Y=np.array(Y))

    X, Y = np.array(X), np.array(Y)
    np.savez("research_data_final.npz",
             X=X, Y=Y,
             X_mean=X.mean(0), X_std=X.std(0)+1e-8,
             Y_mean=Y.mean(0), Y_std=Y.std(0)+1e-8,
             crashed=np.array(crashed))
    logging.info(f"Saved. X shape: {X.shape}, Y shape: {Y.shape}")

if __name__ == "__main__":
    run_collection()