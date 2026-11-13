"""Phase 2b: a neural-network controller trained by evolution strategies (black-box RL).

Policy  u = w . [e, -v, ∫e]  +  MLP([e, v, ∫e, x])        (then saturated to |u| <= U_MAX)
        a learned linear PID part plus a small tanh network for anything PID can't express.

Training is OpenAI-style ES: antithetic Gaussian perturbations of the flat parameter vector,
centred-rank fitness shaping, Adam on the estimated gradient. Every member of the population is
simulated on every training plant *in one vectorised batch*, so a generation costs about as much
as one long simulation. Same scenarios, cost function and held-out test set as phase 2, so the
comparison with Bayesian-optimised PID is apples to apples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.controller_tuning import DT, T_DIST, T_END, TV_WEIGHT, OVERSHOOT_WEIGHT, U_MAX, X_REF, Scenarios

N_IN, HIDDEN = 4, 16


@dataclass
class PolicyShape:
    hidden: int = HIDDEN

    @property
    def sizes(self):
        h = self.hidden
        return {"w_lin": 3, "W1": N_IN * h, "b1": h, "W2": h, "b2": 1}

    @property
    def dim(self):
        return sum(self.sizes.values())

    def unpack(self, theta):
        """theta: (P, dim) -> dict of arrays with a leading population axis."""
        out, i = {}, 0
        h = self.hidden
        shapes = {"w_lin": (3,), "W1": (N_IN, h), "b1": (h,), "W2": (h, 1), "b2": (1,)}
        for k, n in self.sizes.items():
            out[k] = theta[:, i:i + n].reshape((len(theta),) + shapes[k])
            i += n
        return out


# linear gains are learned in log-space-ish units: actual gain = LIN_SCALE * w
LIN_SCALE = np.array([20.0, 10.0, 5.0])
RES_SCALE = 10.0  # network output scale (N)


def simulate_policies(theta, sc: Scenarios, shape=PolicyShape()):
    """Simulate P policies x S scenarios. Returns X, U with shape (steps, P, S)."""
    theta = np.atleast_2d(theta)
    p = shape.unpack(theta)
    P, S = len(theta), sc.n
    steps = int(T_END / DT)
    x, v, integ = np.zeros((P, S)), np.zeros((P, S)), np.zeros((P, S))
    X, U = np.empty((steps, P, S)), np.empty((steps, P, S))
    i_dist = int(T_DIST / DT)
    gains = LIN_SCALE * p["w_lin"]                                   # (P, 3)
    for i in range(steps):
        xm, vm = x + sc.noise_x[i], v + sc.noise_v[i]
        e = X_REF - xm
        lin = gains[:, 0:1] * e - gains[:, 1:2] * vm + gains[:, 2:3] * integ
        feats = np.stack([e, vm, integ, xm], axis=-1)                 # (P, S, 4)
        hid = np.tanh(np.einsum("psi,pih->psh", feats, p["W1"]) + p["b1"][:, None, :])
        res = RES_SCALE * (np.einsum("psh,ph->ps", hid, p["W2"][..., 0]) + p["b2"])
        f = lin + res
        f_sat = np.clip(f, -U_MAX, U_MAX)
        integ += np.where(f == f_sat, e, 0.0) * DT                    # same anti-windup as phase 2
        d = sc.d if i >= i_dist else 0.0
        a = (f_sat + d - sc.b * v - sc.k * x) / sc.m
        v = v + a * DT
        x = x + v * DT
        X[i], U[i] = x, f_sat
    return X, U


def policy_costs(theta, sc: Scenarios, shape=PolicyShape()):
    """Robust cost (mean + 0.5 std over scenarios) for each policy, identical to phase 2's."""
    X, U = simulate_policies(theta, sc, shape)
    t = np.arange(X.shape[0]) * DT
    e = X_REF - X
    iae = np.abs(e).sum(0) * DT
    tv = np.abs(np.diff(U, axis=0)).sum(0)
    over = np.maximum(X[t < T_DIST].max(0) - X_REF, 0) / X_REF * 100
    j = iae + TV_WEIGHT * tv + OVERSHOOT_WEIGHT * over
    j = np.where(np.isfinite(j), j, 1e3)
    return j.mean(1) + 0.5 * j.std(1)


def initial_theta(shape=PolicyShape(), seed=0):
    rng = np.random.default_rng(seed)
    th = rng.normal(0, 0.1, shape.dim)
    th[:3] = [0.3, 0.3, 0.3]           # a weak, stable PID to start from
    th[-1 - shape.hidden:] *= 0.1      # residual network starts near zero
    return th


def train_es(sc: Scenarios, generations=150, pop_half=24, sigma=0.05, lr=0.03, seed=0,
             shape=PolicyShape(), verbose=True):
    rng = np.random.default_rng(seed)
    theta = initial_theta(shape, seed)
    m, v_ = np.zeros_like(theta), np.zeros_like(theta)
    b1, b2 = 0.9, 0.999
    history = []
    best_theta, best_cost = theta.copy(), np.inf
    for g in range(1, generations + 1):
        eps = rng.normal(0, 1, (pop_half, shape.dim))
        cand = np.vstack([theta, theta + sigma * eps, theta - sigma * eps])
        c = policy_costs(cand, sc, shape)
        c0, cp, cn = c[0], c[1:pop_half + 1], c[pop_half + 1:]
        if c0 < best_cost:
            best_cost, best_theta = c0, theta.copy()
        history.append(float(c0))
        ranks = np.argsort(np.argsort(np.r_[cp, cn])) / (2 * pop_half - 1) - 0.5   # centred ranks
        rp, rn = ranks[:pop_half], ranks[pop_half:]
        grad = ((rp - rn)[:, None] * eps).sum(0) / (2 * pop_half * sigma)          # d(cost)/d(theta)
        m = b1 * m + (1 - b1) * grad
        v_ = b2 * v_ + (1 - b2) * grad ** 2
        theta = theta - lr * (m / (1 - b1 ** g)) / (np.sqrt(v_ / (1 - b2 ** g)) + 1e-8)
        if verbose and (g % 10 == 0 or g == 1):
            print(f"  ES gen {g:4d}: cost {c0:.3f}   best {best_cost:.3f}   PID part {np.round(LIN_SCALE * theta[:3], 2)}")
    final = policy_costs(theta[None], sc, shape)[0]
    if final < best_cost:
        best_cost, best_theta = final, theta
    return best_theta, float(best_cost), np.array(history)
