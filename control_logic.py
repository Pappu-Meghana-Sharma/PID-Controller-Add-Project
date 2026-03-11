import numpy as np
from gain_scheduler import GainScheduler

class ResearchController:
    def __init__(self, masses=None, winds=None, gains_grid=None):
        # If no table provided, use a default 1D table (mass only) for backward compatibility
        if masses is None or winds is None or gains_grid is None:
            # Default: mass only (wind = 0 assumed)
            self.scheduler = None
            self.lookup_table = {
                0.027: [12.0, 0.1, 0.05, 0.6, 0.02, 0.01],
                0.045: [15.5, 0.2, 0.08, 0.8, 0.03, 0.02],
                0.060: [18.0, 0.3, 0.10, 1.1, 0.04, 0.03]
            }
        else:
            self.scheduler = GainScheduler(masses, winds, gains_grid)
        self.reset_state()

    def reset_state(self):
        self.integral_angle = np.zeros(3)
        self.integral_rate = np.zeros(3)
        self.prev_angle_error = np.zeros(3)
        self.prev_rate_error = np.zeros(3)

    def get_gains(self, mass, wind_mag=0.0):
        if self.scheduler is not None:
            return self.scheduler.get_gains(mass, wind_mag)
        else:
            # Fallback to old 1D interpolation
            keys = sorted(self.lookup_table.keys())
            m_min, m_max = keys[0], keys[-1]
            m = np.clip(mass, m_min, m_max)
            if m in self.lookup_table:
                return self.lookup_table[m].copy()
            for i in range(len(keys)-1):
                if keys[i] <= m <= keys[i+1]:
                    m0, m1 = keys[i], keys[i+1]
                    g0 = np.array(self.lookup_table[m0])
                    g1 = np.array(self.lookup_table[m1])
                    gains = g0 + (m - m0) * (g1 - g0) / (m1 - m0)
                    return gains.tolist()
            return self.lookup_table[m_min].copy()

    def run_cascaded_control(self, target_angles, curr_angles, curr_rates, gains, dt):
        Kp_a, Ki_a, Kd_a, Kp_r, Ki_r, Kd_r = gains

        # Outer loop: angle error -> desired rate
        angle_error = target_angles - curr_angles
        self.integral_angle += angle_error * dt
        self.integral_angle = np.clip(self.integral_angle, -0.5, 0.5)
        angle_deriv = (angle_error - self.prev_angle_error) / dt

        desired_rates = (Kp_a * angle_error +
                         Ki_a * self.integral_angle +
                         Kd_a * angle_deriv)

        # Inner loop: rate error -> torque
        rate_error = desired_rates - curr_rates
        self.integral_rate += rate_error * dt
        self.integral_rate = np.clip(self.integral_rate, -0.5, 0.5)
        rate_deriv = (rate_error - self.prev_rate_error) / dt

        torque = (Kp_r * rate_error +
                  Ki_r * self.integral_rate +
                  Kd_r * rate_deriv)

        self.prev_angle_error = angle_error.copy()
        self.prev_rate_error = rate_error.copy()

        return torque