"""Minimal Bayesian optimisation: Gaussian-process surrogate + Expected Improvement.

Built only on scikit-learn and scipy so it stays transparent and dependency-free. Parameters
are searched in a box, optionally in log10 space (natural for controller gains).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass
class BOResult:
    x_best: np.ndarray
    y_best: float
    X: np.ndarray                     # every evaluated point (original units)
    y: np.ndarray                     # objective at each point
    best_so_far: np.ndarray = field(init=False)

    def __post_init__(self):
        self.best_so_far = np.minimum.accumulate(self.y)


def expected_improvement(mu, sigma, y_best, xi=0.01):
    """EI for minimisation."""
    sigma = np.maximum(sigma, 1e-12)
    imp = y_best - mu - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


class BayesianOptimizer:
    def __init__(self, bounds, log_scale=True, seed=0):
        self.bounds = np.asarray(bounds, float)
        self.log = log_scale
        self.dim = len(self.bounds)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        lo, hi = (np.log10(self.bounds.T) if log_scale else self.bounds.T)
        self._lo, self._hi = lo, hi

    # unit cube <-> parameter space
    def to_param(self, u):
        z = self._lo + np.asarray(u) * (self._hi - self._lo)
        return 10 ** z if self.log else z

    def to_unit(self, x):
        z = np.log10(x) if self.log else np.asarray(x)
        return (z - self._lo) / (self._hi - self._lo)

    def _gp(self):
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.3] * self.dim,
                                                             length_scale_bounds=(1e-2, 10.0), nu=2.5)
                  + WhiteKernel(1e-4, (1e-8, 1e-1)))
        return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3,
                                        random_state=self.seed)

    def minimize(self, f, n_init=8, n_iter=40, n_candidates=2000, log_objective=True, verbose=False):
        """Minimise f(x) -> float. Uses log(f) for the surrogate when f is positive (smoother)."""
        U = qmc.Sobol(self.dim, seed=self.seed).random(n_init)
        X = [self.to_param(u) for u in U]
        y = [f(x) for x in X]
        U = list(U)
        for i in range(n_iter):
            Y = np.log(np.array(y)) if log_objective else np.array(y)
            gp = self._gp().fit(np.array(U), Y)
            y_best = Y.min()

            def neg_ei(u):
                mu, s = gp.predict(np.atleast_2d(u), return_std=True)
                return -expected_improvement(mu, s, y_best)

            cand = self.rng.random((n_candidates, self.dim))
            ei = -neg_ei(cand)
            starts = cand[np.argsort(ei)[-5:]]
            best_u, best_val = None, np.inf
            for s0 in starts:
                r = minimize(lambda u: neg_ei(u)[0], s0, bounds=[(0, 1)] * self.dim, method="L-BFGS-B")
                if r.fun < best_val:
                    best_u, best_val = r.x, r.fun
            x_new = self.to_param(best_u)
            U.append(best_u)
            X.append(x_new)
            y.append(f(x_new))
            if verbose:
                print(f"  BO iter {i + 1:3d}/{n_iter}: f={y[-1]:.4f}  best={min(y):.4f}")
        X, y = np.array(X), np.array(y)
        j = int(np.argmin(y))
        return BOResult(X[j], float(y[j]), X, y)


def random_search(f, bounds, n_evals, log_scale=True, seed=0) -> BOResult:
    """Baseline with the same evaluation budget, to show BO's sample efficiency."""
    bo = BayesianOptimizer(bounds, log_scale, seed)
    U = np.random.default_rng(seed + 1).random((n_evals, len(bounds)))
    X = np.array([bo.to_param(u) for u in U])
    y = np.array([f(x) for x in X])
    j = int(np.argmin(y))
    return BOResult(X[j], float(y[j]), X, y)
