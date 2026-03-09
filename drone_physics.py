import pybullet as p
import pybullet_data
import numpy as np

class DroneEnvironment:
    def __init__(self,render=False):
        self.physics_client_=p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.drone_id = None
        
    def reset_drone(self,mass=0.027):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.drone_id=p.loadURDF("cf2x.urdf",[0,0,1])
        p.changeDynamics(self.drone_id,-1,mass=mass)
        return self.drone_id
    
    def wind_disturbances(self,force_vector=[0,0,0],torque=[0,0,0]):
        p.applyExternalForce(
            self.drone_id,
            -1,
            forceObj=force_vector,
            posObj=[0,0,0],
            flags=p.WORLD_FRAME
        )
        p.applyExternalForce(
            self.drone_id,
            -1,
            forceObj=torque,
            flags=p.LINK_FRAME
        )
    
    def get_imu_data(self):
        
        pos,quat=p.getBasePositionAndOrientation(self.drone_id)
        angles=p.getEulerQuaternion(quat)
        vel,ang_vel=p.getBaseVelocity(self.drone_id)
        return np.array(angles),np.array(ang_vel)
    
    def step(self):
        p.stepSimulation()
    
    def close(self):
        p.disconnect()