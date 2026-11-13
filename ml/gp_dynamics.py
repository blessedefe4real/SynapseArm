"""Phase 3: learning unmodelled dynamics with a Gaussian process.

A computed-torque controller is only as good as its model. The "true" axis here has things a
datasheet model leaves out:

  * parameter error (heavier payload, more viscous damping, softer spring)
  * Coulomb friction  F_c * tanh(v / v_s)   (discontinuous at v = 0)
  * position-dependent cogging / gravity-like load  A * sin(w x)

The controller runs the trajectory, logs (x, v, a, f), and a GP learns the residual force
r = f - (M a + B v + K x) that the nominal model misses. The GP mean is then added as
feed-forward along the next reference trajectory, episode after episode (iterative learning).
The GP's predictive std gives a confidence band and is used to fade out the correction where
the model hasn't seen data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

M, B, K = 7.0, 1.0, 9.0                         # nominal model
DT, T_END, U_MAX = 1e-3, 20.0, 80.0
KP, KD = 25.0, 10.0                             # servo gains of f' (wn = 5 rad/s, zeta = 1)


@dataclass
class TruePlant:
    m: float = 7.0 * 1.25
    b: float = 1.0 * 2.5
    k: float = 9.0 * 0.85
    coulomb: float = 2.0
    v_stribeck: float = 0.02
    cog_amp: float = 1.5
    cog_freq: float = 4.0

    def nonlinear_force(self, x, v):
        return self.coulomb * np.tanh(v / self.v_stribeck) + self.cog_amp * np.sin(self.cog_freq * x)

    def residual(self, x, v, a):
        """Force the nominal model misses: true dynamics minus nominal model."""
        return (self.m - M) * a + (self.b - B) * v + (self.k - K) * x + self.nonlinear_force(x, v)

    def accel(self, f, x, v):
        return (f - self.b * v - self.k * x - self.nonlinear_force(x, v)) / self.m


def reference(kind="train", t=None):
    """Smooth multi-sine references: x_d, v_d, a_d. 'test' uses frequencies never trained on."""
    t = np.arange(int(T_END / DT)) * DT if t is None else t
    comps = {"train": [(0.5, 0.8, 0.0), (0.3, 2.1, 1.0)],
             "test": [(0.6, 1.3, 0.5), (0.2, 3.0, 2.0)]}[kind]
    x = sum(A * np.sin(w * t + p) for A, w, p in comps)
    v = sum(A * w * np.cos(w * t + p) for A, w, p in comps)
    a = sum(-A * w * w * np.sin(w * t + p) for A, w, p in comps)
    return t, x, v, a


def run(plant: TruePlant, ref, ff=None, plant_aware=False, noise=(1e-4, 2e-3, 0.05), seed=0):
    """Computed-torque tracking. ff: extra feed-forward force along the reference (e.g. GP mean).

    plant_aware=True gives the oracle: the controller knows the true plant exactly.
    Returns log dict with t, x, v, a_meas, f, e.
    """
    t, xd, vd, ad = ref
    n = len(t)
    rng = np.random.default_rng(seed)
    ff = np.zeros(n) if ff is None else ff
    x, v = xd[0], vd[0]
    X, V, A, F = (np.empty(n) for _ in range(4))
    for i in range(n):
        xm, vm = x + rng.normal(0, noise[0]), v + rng.normal(0, noise[1])
        a_cmd = ad[i] + KD * (vd[i] - vm) + KP * (xd[i] - xm)
        if plant_aware:
            f = plant.m * a_cmd + plant.b * vm + plant.k * xm + plant.nonlinear_force(xm, vm)
        else:
            f = M * a_cmd + B * vm + K * xm + ff[i]
        f = float(np.clip(f, -U_MAX, U_MAX))
        a = plant.accel(f, x, v)
        X[i], V[i], A[i], F[i] = x, v, a, f
        v += a * DT
        x += v * DT
    a_meas = A + rng.normal(0, noise[2], n)
    return {"t": t, "x": X, "v": V, "a": a_meas, "f": F, "e": xd - X}


def rmse(log):
    return float(np.sqrt(np.mean(log["e"] ** 2)))


class ResidualGP:
    """GP regression of the unmodelled force r(x, v, a).

    kernel="structured" (default) encodes the physics' additive structure:
        k = linear(x, v, a)  +  RBF(x)  +  RBF(v)  +  noise
    i.e. r ~ (dm a + db v + dk x) + cogging(x) + friction(v). The linear part extrapolates
    parameter error, and the 1-D RBFs can't invent spurious x-v-a interactions, so it generalises
    to trajectories it never saw. kernel="rbf" is the black-box 3-D RBF baseline (scikit-learn).
    """

    def __init__(self, kernel="structured", max_points=600, seed=0):
        assert kernel in ("structured", "rbf")
        self.kernel = kernel
        self.max_points = max_points
        self.rng = np.random.default_rng(seed)
        self.Z, self.r = np.empty((0, 3)), np.empty(0)

    def add(self, log, stride=10):
        Z = np.column_stack([log["x"], log["v"], log["a"]])[::stride]
        r = (log["f"] - (M * log["a"] + B * log["v"] + K * log["x"]))[::stride]
        self.Z, self.r = np.vstack([self.Z, Z]), np.concatenate([self.r, r])
        return self

    # ---- structured kernel (own implementation, hyper-parameters by marginal likelihood)
    @staticmethod
    def _k_struct(theta, A, Bm):
        lin = np.exp(theta[0:3])                      # linear variances per input
        sx, lx, sv, lv = np.exp(theta[3:7])
        Kl = (A * lin) @ Bm.T
        Kx = sx ** 2 * np.exp(-0.5 * (A[:, :1] - Bm[:, :1].T) ** 2 / lx ** 2)
        Kv = sv ** 2 * np.exp(-0.5 * (A[:, 1:2] - Bm[:, 1:2].T) ** 2 / lv ** 2)
        return Kl + Kx + Kv

    def _nll(self, theta, Z, y):
        n = len(y)
        Kmat = self._k_struct(theta, Z, Z) + (np.exp(theta[7]) ** 2 + 1e-8) * np.eye(n)
        try:
            L = np.linalg.cholesky(Kmat)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return 0.5 * y @ alpha + np.log(np.diag(L)).sum() + 0.5 * n * np.log(2 * np.pi)

    def fit(self):
        from scipy.optimize import minimize
        idx = np.arange(len(self.r))
        if len(idx) > self.max_points:
            idx = self.rng.choice(idx, self.max_points, replace=False)
        self.mu, self.sd = self.Z[idx].mean(0), self.Z[idx].std(0) + 1e-9
        Z = (self.Z[idx] - self.mu) / self.sd
        self.y_mu, self.y_sd = self.r[idx].mean(), self.r[idx].std() + 1e-9
        y = (self.r[idx] - self.y_mu) / self.y_sd
        if self.kernel == "rbf":
            kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF([1.0, 0.3, 1.0], (1e-2, 1e2))
                      + WhiteKernel(0.1, (1e-5, 10.0)))
            self.gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, random_state=0).fit(Z, y)
            self.noise_sd = np.sqrt(self.gp.kernel_.k2.noise_level)
            return self
        theta0 = np.log([0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.1, 0.1])
        bounds = [(-9, 3)] * 3 + [(-7, 3), (-4, 3), (-7, 3), (-6, 3), (-8, 1)]
        res = minimize(self._nll, theta0, args=(Z, y), method="L-BFGS-B", bounds=bounds)
        self.theta, self.Ztr = res.x, Z
        Kmat = self._k_struct(res.x, Z, Z) + (np.exp(res.x[7]) ** 2 + 1e-8) * np.eye(len(y))
        self.L = np.linalg.cholesky(Kmat)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        self.noise_sd = np.exp(res.x[7])
        return self

    def hyperparameters(self) -> dict:
        if self.kernel == "rbf":
            return {"kernel": str(self.gp.kernel_)}
        t = np.exp(self.theta)
        return {"linear_var_x_v_a": t[0:3].tolist(), "cogging_amp_len": t[3:5].tolist(),
                "friction_amp_len": t[5:7].tolist(), "noise_sd": float(t[7])}

    def predict(self, x, v, a, return_std=False):
        Zs = (np.column_stack([x, v, a]) - self.mu) / self.sd
        if self.kernel == "rbf":
            m, s = self.gp.predict(Zs, return_std=True)
        else:
            Ks = self._k_struct(self.theta, Zs, self.Ztr)
            m = Ks @ self.alpha
            W = np.linalg.solve(self.L, Ks.T)
            prior = (np.exp(self.theta[0:3]) * Zs ** 2).sum(1) + np.exp(self.theta[3]) ** 2 + np.exp(self.theta[5]) ** 2
            s = np.sqrt(np.maximum(prior - (W ** 2).sum(0), 1e-12))
        m, s = self.y_mu + self.y_sd * m, self.y_sd * s
        return (m, s) if return_std else m

    def feedforward(self, ref, confidence_fade=True):
        """GP mean along the reference; faded where the GP is unsure (std >> noise level)."""
        _, xd, vd, ad = ref
        mean, std = self.predict(xd, vd, ad, return_std=True)
        if not confidence_fade:
            return mean
        noise = self.noise_sd * self.y_sd
        return mean / (1.0 + (std / (3 * noise + 1e-9)) ** 2)
