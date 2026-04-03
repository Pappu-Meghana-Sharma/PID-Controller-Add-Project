import numpy as np
def traj_hover(t):
    if t < 2.0:
        z  = 0.05 + (0.5 - 0.05) * (t / 2.0)
        vz = (0.5 - 0.05) / 2.0
        return np.array([0.0, 0.0, z]), np.array([0.0, 0.0, vz])
    return np.array([0.0, 0.0, 0.5]), np.zeros(3)


def traj_circle(t):
    r = 0.5
    RAMP_END  = 3.0   
    BLEND_END = 4.5   

    if t < RAMP_END:
        alpha = t / RAMP_END
        z  = 0.05 + (0.5 - 0.05) * alpha
        vz = (0.5 - 0.05) / RAMP_END
        return np.array([0.0, 0.0, z]), np.array([0.0, 0.0, vz])

    t_c = t - RAMP_END                     
    angle  = t_c - np.pi / 2
    cx, cy = r * np.cos(angle), r + r * np.sin(angle)   
    vx     = -r * np.sin(angle)
    vy     =  r * np.cos(angle)

    if t < BLEND_END:
        blend = (t - RAMP_END) / (BLEND_END - RAMP_END)   
        blend = blend * blend * (3 - 2 * blend)           
        pos = np.array([blend * cx, blend * cy, 0.5])
        vel = np.array([blend * vx, blend * vy, 0.0])
        return pos, vel

    return np.array([cx, cy, 0.5]), np.array([vx, vy, 0.0])

# def traj_circle(t):
#     r = 0.5
#     if t < 3.5:
#         alpha = t / 3.0
#         z  = 0.05 + (0.5 - 0.05) * alpha
#         vz = (0.5 - 0.05) / 3.0
#         return np.array([0.0, 0.0, z]), np.array([0.0, 0.0, vz])
#     t_adj = t - 3.5
#     return (np.array([ r*np.cos(t_adj),  r*np.sin(t_adj), 0.5]),
#             np.array([-r*np.sin(t_adj),  r*np.cos(t_adj), 0.0]))

def traj_figure8(t):
    r = 0.4
    if t < 3.0:
        alpha = t / 3.0
        z  = 0.05 + (0.5 - 0.05) * alpha
        vz = (0.5 - 0.05) / 3.0
        return np.array([0.0, 0.0, z]), np.array([0.0, 0.0, vz])
    t_adj = t - 3.0
    return (np.array([r*np.sin(t_adj),
                      r*np.sin(t_adj)*np.cos(t_adj), 0.5]),
            np.array([r*np.cos(t_adj),
                      r*(np.cos(t_adj)**2 - np.sin(t_adj)**2), 0.0]))