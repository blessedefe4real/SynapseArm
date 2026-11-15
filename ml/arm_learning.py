"""Learning on the full 6-DOF arm: learned feed-forward and evolution-strategies gain tuning.

1. Learned feed-forward (iterative learning on one arm)
   The computed-torque controller only knows the rigid-body model. After one run, the momentum
   observer (fed by Kalman-filtered velocity) gives the torque the model missed, without any
   acceleration measurement. Its de-filtered version, -(r + r_dot / K), is what the motors had
   to supply on top of the model. Per joint we learn it with a structured model:
       physics mean   w . [q_dot_j, tanh(q_dot_j / v_s), payload J^T g, 1]   (ridge)
     + GP on q_dot_j  for what the basis can't express (Stribeck static friction)
   and add it as feed-forward along the *next* reference, a different trajectory.

2. Servo-gain tuning by evolution strategies
   12 parameters (kp, kd per joint, log-space) tuned across a fleet of arms with unit-specific
   friction and payload, trading tracking error against torque chatter (velocity noise times kd
   times inertia). Same OpenAI-ES recipe as the 1-axis controller (antithetic sampling,
   centred ranks, Adam); every candidate is simulated on every training arm in one batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from ml.fault_detection import (DT, K_OBS, KD, KP, V_STRIBECK, FleetRun, Units, kalman_velocity,
                                momentum_observer, payload_gravity_basis, random_moves, simulate_fleet)
from robotics.dynamics import ArmDynamics


# ----------------------------------------------------------------------------- ground truth
def true_unmodelled_torque(units: Units, dyn: ArmDynamics, q, qd):
    """What the motors must add to the rigid-body model on these arms: friction (incl. Stribeck)
    minus the payload deviation's gravity torque. Along a reference: the ideal feed-forward."""
    vs = 0.02
    st = units.strib * np.exp(-(qd / V_STRIBECK) ** 2)
    fric = units.visc * qd + (units.coul + st) * np.tanh(qd / vs)
    run = FleetRun(q, qd, None, None, None, None, None, None)
    g = payload_gravity_basis(run, dyn)                       # torque per kg of extra payload
    return fric - units.payload_dm[:, None] * g


# ----------------------------------------------------------------------------- learned feed-forward
def unmodelled_torque_target(run: FleetRun, dyn: ArmDynamics):
    """Observer-based estimate of the torque the model missed (no acceleration needed)."""
    kf = kalman_velocity(run)
    r = momentum_observer(kf, dyn)
    return kf, -(r + np.gradient(r, DT, axis=0) / K_OBS)


def _basis(q, qd, dyn):
    run = FleetRun(q, qd, None, None, None, None, None, None)
    g = payload_gravity_basis(run, dyn)
    return np.stack([qd, np.tanh(qd / 0.02), g, np.ones_like(qd)], axis=-1)   # (steps, N, 6, 4)


@dataclass
class ArmFeedForward:
    """Per arm, per joint: ridge physics mean + GP on joint velocity (structured residual model)."""
    dyn: ArmDynamics
    ridge: float = 1e-3
    gp_points: int = 250
    use_gp: bool = True

    def fit(self, run: FleetRun, warmup=100, seed=0):
        kf, y = unmodelled_torque_target(run, self.dyn)
        B = _basis(kf.q, kf.qd, self.dyn)[warmup:]
        y = y[warmup:]
        steps, N = y.shape[:2]
        scale = np.sqrt((B ** 2).mean(0)) + 1e-9
        Bn = B / scale
        A = np.einsum("tnjk,tnjl->njkl", Bn, Bn) + self.ridge * steps * np.eye(4)
        self.w = np.linalg.solve(A, np.einsum("tnjk,tnj->njk", Bn, y)[..., None])[..., 0] / scale
        self.gps = {}
        if self.use_gp:
            rng = np.random.default_rng(seed)
            resid = y - np.einsum("tnjk,njk->tnj", B, self.w)
            qd = kf.qd[warmup:]
            for n in range(N):
                idx = rng.choice(steps, min(self.gp_points, steps), replace=False)
                for j in range(6):
                    k = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.1, (0.01, 2.0)) + WhiteKernel(0.5, (1e-4, 50)))
                    gp = GaussianProcessRegressor(k, normalize_y=True, random_state=0)
                    self.gps[n, j] = gp.fit(qd[idx, n, j:j + 1], resid[idx, n, j])
        return self

    def predict(self, q, qd):
        """Feed-forward torque along a reference (steps, N, 6)."""
        B = _basis(q, qd, self.dyn)
        out = np.einsum("tnjk,njk->tnj", B, self.w)
        for (n, j), gp in self.gps.items():
            out[:, n, j] += gp.predict(qd[:, n, j:j + 1])
        return out


def tracking_rms(run: FleetRun):
    """Per-run, per-joint RMS tracking error (rad), using the true (noise-free) joint position."""
    return np.sqrt(((run.q_des - run.q) ** 2).mean(0))


# ----------------------------------------------------------------------------- ES gain tuning
WEIGHT_JOINT = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.3])   # tool-position relevance of each joint
CHATTER_PENALTY = 20.0


def simulate_gains(log_gains, units: Units, seed, ff_model=None, t_run=3.0):
    """Every candidate gain set (P, 12) on every arm, one batched simulation.
    Returns weighted tracking error (mrad) and torque chatter (mean |d tau/dt|, N m/s) per candidate."""
    G = np.atleast_2d(log_gains)
    P, A = len(G), len(units.payload_dm)
    kp = np.repeat(np.exp(G[:, :6]), A, axis=0)
    kd = np.repeat(np.exp(G[:, 6:]), A, axis=0)
    U = Units(*(np.tile(getattr(units, f), (P, 1)) if getattr(units, f).ndim == 2 else np.tile(getattr(units, f), P)
                for f in ("visc", "coul", "strib", "payload_dm")))
    ref = random_moves(A, np.random.default_rng(seed), ArmDynamics().robot, t_run)
    ff = None if ff_model is None else np.tile(ff_model.predict(ref[0], ref[1]), (1, P, 1))
    ref = tuple(np.tile(x, (1, P, 1)) for x in ref)
    run = simulate_fleet(P * A, seed=seed, units=U, kp=kp, kd=kd, ref=ref, t_run=t_run, ff=ff, noise_groups=P,
                       noise_seed=seed + 1)
    err = np.sqrt(((run.q_des - run.q) ** 2).mean(0)) * 1e3                    # (P*A, 6) mrad
    track = (err * WEIGHT_JOINT).sum(1).reshape(P, A).mean(1)
    chatter = np.abs(np.diff(run.tau, axis=0) / DT).mean((0, 2)).reshape(P, A).mean(1)
    return track, chatter


def default_log_gains():
    return np.r_[np.full(6, np.log(KP)), np.full(6, np.log(KD))]


def train_gains_es(units: Units, ff_model=None, generations=25, pop_half=4, sigma=0.15, lr=0.08, seed=0,
                   verbose=True):
    """Best tracking (with the learned feed-forward in the loop) without more torque chatter than
    the default gains: cost = tracking + penalty * max(0, chatter / chatter_default - 1)."""
    rng = np.random.default_rng(seed)
    theta = default_log_gains()
    _, ch0 = simulate_gains(np.vstack([theta, theta]), units, seed=999, ff_model=ff_model)
    chatter_ref = float(ch0[0])
    m, v = np.zeros(12), np.zeros(12)
    hist = []
    for g in range(1, generations + 1):
        eps = rng.normal(0, 1, (pop_half, 12))
        cand = np.vstack([theta, theta + sigma * eps, theta - sigma * eps])
        tr, ch = simulate_gains(cand, units, seed=1000 + g, ff_model=ff_model)   # fresh trajectories
        c = tr + CHATTER_PENALTY * np.maximum(0, ch / chatter_ref - 1)
        hist.append((float(c[0]), float(tr[0]), float(ch[0])))
        cp, cn = c[1:pop_half + 1], c[pop_half + 1:]
        ranks = np.argsort(np.argsort(np.r_[cp, cn])) / (2 * pop_half - 1) - 0.5
        grad = ((ranks[:pop_half] - ranks[pop_half:])[:, None] * eps).sum(0) / (2 * pop_half * sigma)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        theta = theta - lr * (m / (1 - 0.9 ** g)) / (np.sqrt(v / (1 - 0.999 ** g)) + 1e-8)
        theta[:6] = np.clip(theta[:6], np.log(25), np.log(4000))
        theta[6:] = np.clip(theta[6:], np.log(5), np.log(150))
        if verbose:
            print(f"  ES gen {g:3d}: track {tr[0]:.3f} mrad  chatter {ch[0] / chatter_ref:.2f}x  "
                  f"kp {np.exp(theta[:6]).round(0)}  kd {np.exp(theta[6:]).round(1)}")
    return theta, chatter_ref, np.array(hist)
