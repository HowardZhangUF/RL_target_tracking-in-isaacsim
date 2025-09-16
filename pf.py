# pf.py
import numpy as np

class ParticleFilter2D:
    """Constant-velocity PF over relative state [x, y, vx, vy]."""

    def __init__(self, N=600, proc_pos_std=0.06, proc_vel_std=0.5, meas_std=0.05, resample_frac=0.5, rng=None):
        self.N = int(N)
        self.proc_pos_std = float(proc_pos_std)  # m per sqrt(s)
        self.proc_vel_std = float(proc_vel_std)  # m/s per sqrt(s)
        self.meas_std = float(meas_std)          # m
        self.resample_frac = float(resample_frac)
        self.rng = np.random.default_rng() if rng is None else rng
        self.p = None  # (N,4) particles
        self.w = None  # (N,)  weights

    def init(self, mean_xy, std_pos=0.2, std_vel=0.2):
        mean_xy = np.asarray(mean_xy, dtype=np.float32)
        self.p = np.zeros((self.N, 4), dtype=np.float32)
        self.p[:, 0] = mean_xy[0] + self.rng.normal(0, std_pos, size=self.N)
        self.p[:, 1] = mean_xy[1] + self.rng.normal(0, std_pos, size=self.N)
        self.p[:, 2] = self.rng.normal(0, std_vel, size=self.N)
        self.p[:, 3] = self.rng.normal(0, std_vel, size=self.N)
        self.w = np.full(self.N, 1.0 / self.N, dtype=np.float32)

    def predict(self, dt):
        if self.p is None:
            raise RuntimeError("PF not initialized via .init()")
        dt = float(dt)
        sdt = max(dt, 1e-6) ** 0.5
        self.p[:, 0] += self.p[:, 2] * dt + self.rng.normal(0, self.proc_pos_std * sdt, size=self.N)
        self.p[:, 1] += self.p[:, 3] * dt + self.rng.normal(0, self.proc_pos_std * sdt, size=self.N)
        self.p[:, 2] += self.rng.normal(0, self.proc_vel_std * sdt, size=self.N)
        self.p[:, 3] += self.rng.normal(0, self.proc_vel_std * sdt, size=self.N)

    def update(self, z_xy, was_occluded=False):
        """z_xy: np.array([dx, dy]) or None when occluded.
        was_occluded: True if previous observation was None (optional)"""
        if z_xy is None:
            return  # no measurement → keep weights
        
        z = np.asarray(z_xy, dtype=np.float32)
        diff = self.p[:, :2] - z[None, :]
        sq_dist = np.sum(diff * diff, axis=1)
        
        # Check if observation is far from ALL particles
        min_sq_dist = np.min(sq_dist)
        
        # If target reappeared far from all particles, partially reinitialize
        if was_occluded and min_sq_dist > 1.0:  # Threshold distance in squared meters
            # Reinitialize some percentage of particles around new observation
            reinit_frac = 0.5  # 50% of particles
            n_reinit = int(self.N * reinit_frac)
            
            # Choose particles with lowest weights to replace
            indices = np.argsort(self.w)[:n_reinit]
            
            # Reset their positions and velocities
            self.p[indices, 0] = z[0] + self.rng.normal(0, 0.05, size=n_reinit)
            self.p[indices, 1] = z[1] + self.rng.normal(0, 0.05, size=n_reinit)
            self.p[indices, 2:4] = self.rng.normal(0, 0.2, size=(n_reinit, 2))
        
        # Continue with regular update...
        inv_2s2 = 1.0 / (2.0 * self.meas_std * self.meas_std + 1e-12)
        ll = np.exp(-sq_dist * inv_2s2).astype(np.float64)
        self.w = (self.w * ll)
        sw = float(self.w.sum())
        if sw <= 0 or not np.isfinite(sw):
            self.w[:] = 1.0 / self.N
        else:
            self.w = (self.w / sw).astype(np.float32)
        # resample if degenerate
        neff = 1.0 / float(np.sum(self.w * self.w) + 1e-12)
        if neff < self.resample_frac * self.N:
            self._systematic_resample()

    def _systematic_resample(self):
        N = self.N
        positions = (self.rng.random() + np.arange(N)) / N
        cumsum = np.cumsum(self.w)
        cumsum[-1] = 1.0
        idx = np.searchsorted(cumsum, positions)
        self.p = self.p[idx]
        self.w.fill(1.0 / N)

    def estimate(self):
        """Return (mean_state[4], cov_xy[2x2], N_eff)."""
        if self.p is None:
            return None, None, 0.0
        W = self.w[:, None]
        mean = np.sum(self.p * W, axis=0).astype(np.float32)
        #xy = self.p[:, :2]
        diff = (self.p - mean[None, :])
        cov = (diff.T * self.w).dot(diff).astype(np.float32)   # 4x4
        neff = 1.0 / float(np.sum(self.w * self.w) + 1e-12)
        return mean, cov, neff
