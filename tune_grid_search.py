from drone_physics import DroneEnvironment
import numpy as np
import pybullet as p
def tune_mass(mass):
    best_itae=float('inf')
    best_kp=0
    kp_candidates=np.linspace(5,25,20)
    
    for kp in kp_candidates:
        env=DroneEnvironment(render=False)
        drone=env.reset_drone(mass=mass)
        itae=0
        target=0.1
        
        for t in range(240):
            angles,rates=env.get_imu_data()
            error=target-angles[1]
            torque=kp*error-0.01*rates[1]
            p.applyExternalTorque(drone,-1,[0,torque,0],p.LINK_FRAME)
            env.step()
            itae+=(t/240.0)*abs(error)
            
        
        if itae<best_itae:
            best_itae=itae
            best_kp=kp
        env.close()
        
    return best_kp