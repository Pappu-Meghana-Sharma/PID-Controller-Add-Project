import numpy as np

class GainScheduler:
    """
    2D gain scheduler using bilinear interpolation on a grid of (mass, wind_magnitude).
    The table is defined by:
        - masses: list of mass breakpoints
        - winds: list of wind magnitude breakpoints
        - gains_grid: 3D array of shape (len(masses), len(winds), 6)
          where the last dimension holds [Kp_a, Ki_a, Kd_a, Kp_r, Ki_r, Kd_r]
    """
    def __init__(self, masses, winds, gains_grid):
        self.masses = np.array(masses)
        self.winds = np.array(winds)
        self.gains_grid = np.array(gains_grid)  # shape (M, W, 6)

    def get_gains(self, mass, wind_mag):
        # Clip to table range
        m = np.clip(mass, self.masses[0], self.masses[-1])
        w = np.clip(wind_mag, self.winds[0], self.winds[-1])

        # Find surrounding indices
        i = np.searchsorted(self.masses, m) - 1
        i = max(0, min(i, len(self.masses)-2))
        j = np.searchsorted(self.winds, w) - 1
        j = max(0, min(j, len(self.winds)-2))

        m0, m1 = self.masses[i], self.masses[i+1]
        w0, w1 = self.winds[j], self.winds[j+1]

        # Gains at the four corners
        g00 = self.gains_grid[i, j]
        g01 = self.gains_grid[i, j+1]
        g10 = self.gains_grid[i+1, j]
        g11 = self.gains_grid[i+1, j+1]

        # Bilinear interpolation
        if m1 == m0:
            alpha_m = 0.5
        else:
            alpha_m = (m - m0) / (m1 - m0)

        if w1 == w0:
            alpha_w = 0.5
        else:
            alpha_w = (w - w0) / (w1 - w0)

        # Interpolate in mass first, then in wind
        g0 = g00 + alpha_m * (g10 - g00)
        g1 = g01 + alpha_m * (g11 - g01)
        gains = g0 + alpha_w * (g1 - g0)

        return gains.tolist()