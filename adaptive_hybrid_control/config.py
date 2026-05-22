import numpy as np

# Simulation Frequencies
SIM_FREQ = 240
CTRL_FREQ = 48
CTRL_DT = 1.0 / CTRL_FREQ

# Adaptive MPC Frequency Bounds (Hz)
MPC_FREQ_MIN = 12
MPC_FREQ_MAX = 48
MPC_FREQ_NOM = 24

MPC_EVERY_MIN = CTRL_FREQ // MPC_FREQ_MAX  # = 1
MPC_EVERY_MAX = CTRL_FREQ // MPC_FREQ_MIN  # = 4
MPC_EVERY_NOM = CTRL_FREQ // MPC_FREQ_NOM  # = 2

# Simulation Configuration
DURATION = 20.0  # Flight duration in seconds
BASE_MASS = 0.027  # Crazyflie 2.x empty mass (kg)
PAYLOAD_MASS = 0.015  # Attachment mass (kg)
GRAVITY = 9.81  # m/s^2

# Actuator Limits
MIN_RPM = 0
MAX_RPM = 21702

# Motor Parameters
KF = 3.16e-10  # Thrust coefficient
KM = 7.94e-12  # Torque coefficient
ARM = 0.0397   # Quadcopter arm length (m)

# Inertia Matrix Terms
Ixx = 1.4e-5
Iyy = 1.4e-5
Izz = 2.17e-5

# Hover bias reference
HOVER_RPM_BASE = np.sqrt(BASE_MASS * GRAVITY / (4.0 * KF))

# Disturbances
WIND_NONE = np.array([0.0, 0.0, 0.0])
WIND_LIGHT = np.array([0.0010, 0.0010, 0.0])
WIND_MEDIUM = np.array([0.0035, 0.0035, 0.0])

# Initialization
INIT_XYZS = np.array([[0.0, 0.0, 0.05]])
INIT_RPYS = np.array([[0.0, 0.0, 0.0]])
PRINT_HZ  = 4

# Motor Mixing constants
_S2 = np.sqrt(2.0)
_AX = _S2 / (4.0 * ARM)
_AZ = KF / (4.0 * KM)

# Motor allocation matrix
ALLOC = np.array([
    [0.25, -_AX, -_AX, -_AZ],
    [0.25,  _AX, -_AX,  _AZ],
    [0.25,  _AX,  _AX, -_AZ],
    [0.25, -_AX,  _AX,  _AZ],
])
