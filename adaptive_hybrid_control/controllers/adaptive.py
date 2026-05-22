import numpy as np
from ..config import CTRL_FREQ, MPC_EVERY_NOM, MPC_EVERY_MIN, MPC_EVERY_MAX

class LyapunovMonitor:
    """
    Computes a quadratic Lyapunov candidate V(x) = xᵀ P x where
    x = [pos_error; vel_error] ∈ R⁶ and P is the LQR terminal cost matrix
    from the MPC (already computed via DARE).

    The stability margin σ(t) = -dV/dt / (V(t) + ε) is used to:
      - Quantify how quickly the energy is dissipating (σ > 0 → converging)
      - Drive the adaptive scheduler and adaptive gains

    Stability classification:
      σ > STABLE_THRESH    → "stable"    (energy decaying rapidly)
      σ > MARGINAL_THRESH  → "marginal"  (slow convergence)
      σ ≤ MARGINAL_THRESH  → "unstable"  (energy growing)
    """
    STABLE_THRESH   = 0.5     # σ threshold for "stable"
    MARGINAL_THRESH = 0.05    # σ threshold for "marginal"

    def __init__(self, P_matrix):
        self.P      = P_matrix          # (6×6) positive-definite
        self.V_prev = None
        self.t_prev = None
        self.sigma  = 0.0               # stability margin
        self.V      = 0.0
        self.dVdt   = 0.0

        # History for logging / plotting
        self.V_history     = []
        self.sigma_history = []
        self.mpc_hz_history = []

    def update(self, cur_pos, cur_vel, ref_pos, ref_vel, t):
        """Call once per control step. Returns stability margin σ."""
        err   = np.hstack([cur_pos - ref_pos, cur_vel - ref_vel])
        V     = float(err @ self.P @ err)
        self.V = V

        if self.V_prev is not None and self.t_prev is not None:
            dt = t - self.t_prev
            if dt > 1e-9:
                self.dVdt = (V - self.V_prev) / dt
            else:
                self.dVdt = 0.0
        else:
            self.dVdt = 0.0

        eps   = 1e-6
        self.sigma = -self.dVdt / (V + eps)

        self.V_prev = V
        self.t_prev = t

        self.V_history.append(V)
        self.sigma_history.append(self.sigma)
        return self.sigma

    @property
    def stability_label(self):
        if self.sigma > self.STABLE_THRESH:
            return "STABLE"
        elif self.sigma > self.MARGINAL_THRESH:
            return "MARGINAL"
        else:
            return "UNSTABLE"


class AdaptiveScheduler:
    """
    Adjusts MPC execution period (MPC_EVERY) based on Lyapunov stability.

    Logic:
      - σ > STABLE_THRESH  → system is converging fast; relax MPC (increase period)
      - σ < MARGINAL_THRESH → energy growing; tighten MPC (decrease period)
      - Between thresholds   → hold current period

    A hysteresis counter prevents rapid chattering between levels.
    """
    STABLE_THRESH   = LyapunovMonitor.STABLE_THRESH
    MARGINAL_THRESH = LyapunovMonitor.MARGINAL_THRESH
    HYSTERESIS      = 10    # steps before changing mode

    def __init__(self):
        self.mpc_every    = MPC_EVERY_NOM
        self._hold_count  = 0

    def update(self, sigma):
        """Returns current mpc_every (steps between MPC solves)."""
        self._hold_count += 1
        if self._hold_count < self.HYSTERESIS:
            return self.mpc_every

        if sigma > self.STABLE_THRESH:
            # Stable → can afford slower MPC
            new_every = min(self.mpc_every + 1, MPC_EVERY_MAX)
        elif sigma < self.MARGINAL_THRESH:
            # Unstable → need faster MPC
            new_every = max(self.mpc_every - 1, MPC_EVERY_MIN)
        else:
            new_every = self.mpc_every

        if new_every != self.mpc_every:
            self.mpc_every   = new_every
            self._hold_count = 0   # reset hysteresis after change

        return self.mpc_every

    @property
    def current_mpc_hz(self):
        return CTRL_FREQ / self.mpc_every


class AdaptiveGainScheduler:
    """
    Scales PID gains based on Lyapunov stability margin σ and position error.

    Gain law:
        scale = clip(1 + α*(1 - σ_norm) + β*err_norm, min_scale, max_scale)

    where:
        σ_norm   = clip(σ / STABLE_THRESH, 0, 1)    ← 0 = unstable, 1 = stable
        err_norm = clip(|pos_error| / ERR_REF, 0, 1) ← 0 = no error, 1 = large
        α, β     = tuning coefficients

    When σ is low (unstable) or error is large → scale > 1 → more aggressive Kp.
    When σ is high (stable) and error is small → scale ≈ 1 → nominal gains.

    The scaled gains are passed to DSLPIDControl at each step by directly
    overwriting the internal gain arrays (safe, they are numpy scalars).
    """
    NOM_KP_XY  = 0.4
    NOM_KI_XY  = 0.05
    NOM_KD_XY  = 0.2
    NOM_KP_Z   = 1.25
    NOM_KI_Z   = 0.05
    NOM_KD_Z   = 0.4

    ALPHA       = 0.6    # gain boost from instability
    BETA        = 0.3    # gain boost from position error
    ERR_REF     = 0.5    # position error (m) that gives full β contribution
    MIN_SCALE   = 0.7    # never reduce gains below 70% of nominal
    MAX_SCALE   = 2.0    # never exceed 2× nominal (prevent oscillation)

    def __init__(self, pid_ctrl):
        self.pid   = pid_ctrl
        self._scale = 1.0
        self._read_nominal_gains()

    def _read_nominal_gains(self):
        """Try to read gains from DSLPIDControl; fall back to our constants."""
        try:
            self.nom_P_xy = self.pid.P_COEFF_FOR.copy()
            self.nom_I_xy = self.pid.I_COEFF_FOR.copy()
            self.nom_D_xy = self.pid.D_COEFF_FOR.copy()
        except AttributeError:
            # Fallback: set as arrays matching DSLPIDControl's structure
            self.nom_P_xy = np.array([self.NOM_KP_XY, self.NOM_KP_XY, self.NOM_KP_Z])
            self.nom_I_xy = np.array([self.NOM_KI_XY, self.NOM_KI_XY, self.NOM_KI_Z])
            self.nom_D_xy = np.array([self.NOM_KD_XY, self.NOM_KD_XY, self.NOM_KD_Z])

    def update(self, sigma, pos_error_norm):
        """Compute and apply new scaled gains. Returns current scale factor."""
        sigma_norm  = float(np.clip(sigma / LyapunovMonitor.STABLE_THRESH, 0.0, 1.0))
        err_norm    = float(np.clip(pos_error_norm / self.ERR_REF, 0.0, 1.0))

        scale = 1.0 + self.ALPHA * (1.0 - sigma_norm) + self.BETA * err_norm
        scale = float(np.clip(scale, self.MIN_SCALE, self.MAX_SCALE))
        self._scale = scale

        # Apply scaled gains to PID controller
        try:
            self.pid.P_COEFF_FOR = self.nom_P_xy * scale
            self.pid.I_COEFF_FOR = self.nom_I_xy * np.clip(scale, 0.8, 1.5)  # dampen I scaling
            self.pid.D_COEFF_FOR = self.nom_D_xy * np.clip(scale, 0.9, 1.3)  # D needs less boost
        except AttributeError:
            pass

        return scale

    @property
    def scale(self):
        return self._scale
