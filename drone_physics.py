import pybullet as p
import pybullet_data
import numpy as np

class DroneEnvironment:
    def __init__(self, render=False):
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.drone_id = None
        self.obstacle_ids = []
        self.payload_id = None
        self.wind_force = [0, 0, 0]
        self.wind_torque = [0, 0, 0]
        self.dt = 1/240.0   #Pybullet default time step

    def reset_drone(self, mass=0.027):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.drone_id = p.loadURDF("cf2x.urdf", [0, 0, 1])
        p.changeDynamics(self.drone_id, -1, mass=mass)
        self.obstacle_ids = []
        self.payload_id = None
        return self.drone_id

    def add_obstacle(self, urdf_path, position, orientation=[0, 0, 0, 1]):
        obs_id = p.loadURDF(urdf_path, position, orientation, useFixedBase=True)
        self.obstacle_ids.append(obs_id)
        return obs_id

    def remove_obstacle(self, obstacle_id):
        if obstacle_id in self.obstacle_ids:
            p.removeBody(obstacle_id)
            self.obstacle_ids.remove(obstacle_id)

    def clear_obstacles(self):
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id)
        self.obstacle_ids = []

    def get_distance_to_obstacles(self):
        if not self.obstacle_ids:
            return []
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        distances = []
        for obs_id in self.obstacle_ids:
            obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
            dist = np.linalg.norm(np.array(drone_pos) - np.array(obs_pos))
            distances.append(dist)
        return distances

    def set_payload_mass(self, mass):
        p.changeDynamics(self.drone_id, -1, mass=mass)

    def attach_payload_body(self, urdf_path, position_offset, mass=None):
        payload = p.loadURDF(urdf_path, [0, 0, 0])
        if mass is not None:
            p.changeDynamics(payload, -1, mass=mass)
        p.createConstraint(
            self.drone_id, -1,
            payload, -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            position_offset
        )
        self.payload_id = payload
        return payload

    def remove_payload(self):
        if self.payload_id is not None:
            p.removeBody(self.payload_id)
            self.payload_id = None

    def get_payload_mass(self):
        if self.payload_id is None:
            return 0.0
        dyn_info = p.getDynamicsInfo(self.payload_id, -1)
        return dyn_info[0]

    def get_mass(self):
        base_mass = p.getDynamicsInfo(self.drone_id, -1)[0]
        return base_mass + self.get_payload_mass()

    def set_wind(self, force=[0,0,0], torque=[0,0,0]):
        self.wind_force = force
        self.wind_torque = torque

    def apply_wind(self):
        p.applyExternalForce(self.drone_id, -1, self.wind_force, [0,0,0], p.WORLD_FRAME)
        p.applyExternalTorque(self.drone_id, -1, self.wind_torque, p.LINK_FRAME)

    def wind_disturbances(self, force=[0,0,0], torque=[0,0,0]):
        p.applyExternalForce(self.drone_id, -1, force, [0,0,0], p.WORLD_FRAME)
        p.applyExternalTorque(self.drone_id, -1, torque, p.LINK_FRAME)

    def get_imu_data(self):
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        angles = p.getEulerFromQuaternion(quat)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        return np.array(angles), np.array(ang_vel)

    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        return np.array(pos)

    def get_linear_velocity(self):
        vel, _ = p.getBaseVelocity(self.drone_id)
        return np.array(vel)

    def get_angular_velocity(self):
        _, ang_vel = p.getBaseVelocity(self.drone_id)
        return np.array(ang_vel)

    def apply_control(self, thrust, torques):
        force = [0, 0, thrust]
        p.applyExternalForce(self.drone_id, -1, force, [0,0,0], p.LINK_FRAME)
        p.applyExternalTorque(self.drone_id, -1, torques, p.LINK_FRAME)

    def step(self):
        p.stepSimulation()

    def close(self):
        p.disconnect()