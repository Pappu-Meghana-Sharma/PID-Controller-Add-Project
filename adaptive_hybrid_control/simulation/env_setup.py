import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from ..config import SIM_FREQ, CTRL_FREQ, INIT_XYZS, INIT_RPYS, BASE_MASS

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

def initialize_env(gui=True):
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=INIT_XYZS.copy(),
        initial_rpys=INIT_RPYS.copy(),
        physics=Physics.PYB,
        pyb_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=gui,
        record=False,
    )
    
    pyb_client = env.getPyBulletClient()
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-50,
        cameraTargetPosition=[0.0, 0.0, 0.5],
        physicsClientId=pyb_client
    )
    
    obs, info = env.reset()
    p.changeDynamics(env.DRONE_IDS[0], -1, mass=BASE_MASS, physicsClientId=pyb_client)
    
    return env, pyb_client, obs, info
