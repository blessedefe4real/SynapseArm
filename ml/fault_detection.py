"""Collision and fault detection on joint torques: classical observer vs physics + learning.

Fleet simulation
    Each simulated arm has its own friction (+/-30% unit-to-unit) and payload (2 +/- 0.5 kg) and
    runs random smooth point-to-point moves under computed-torque control. The controller and the
    detectors only know the *rigid-body* model (no friction model), like most deployed arms.

Faults (onset at a random time)
    collision  - half-sine contact force of 40-150 N on the tool, 50-300 ms
    payload    - unexpected extra 1.5-4 kg picked up (quasi-static wrench)
    wear       - one joint's friction rises (viscous x3, Coulomb x2), e.g. lubrication loss
    actuator   - one joint's motor delivers only 60-80% of commanded torque

Detectors (all from measured q, qd and commanded torque only)
    naive      - |tau_cmd - ID(q, qd, qdd_desired)|: inverse-dynamics residual
    observer   - generalised-momentum observer (De Luca): r -> external/unmodelled torque,
                 no acceleration measurement needed
    learned    - observer residual minus a gradient-boosted model of the *normal* residual
                 (learned fleet friction + model error), trained on fault-free runs only
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotics.dynamics import G, ArmDynamics

DT = 2e-3
T_RUN = 6.0
FAULTS = ("collision", "payload", "wear", "actuator")
KP, KD = 400.0, 40.0            # computed-torque servo (wn = 20 rad/s, zeta = 1)
K_OBS = 50.0                    # observer bandwidth (1/s)


# --------------------------------------------------------------------------- kinematic helpers
def flange_point_jacobian(dyn: ArmDynamics, q, offset_m=(0.0, 0.0, 0.1)):
    """Batched linear Jacobian (N, 3, 6) and position (N, 3) of a point fixed in the flange frame (m)."""
    q = np.atleast_2d(q)
    N = q.shape[0]
    R, ps = dyn._link_rot_pos(q)
    Rc = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    p = np.zeros((N, 3))
    z_axes, origins = [], []
    for i in range(dyn.n):
        z_axes.append(Rc[:, :, 2].copy())
        origins.append(p.copy())
        p = p + np.einsum("nij,nj->ni", Rc @ R[:, i], ps[:, i])
        Rc = Rc @ R[:, i]
    pt = p + np.einsum("nij,j->ni", Rc, np.asarray(offset_m))
    J = np.stack([np.cross(z, pt - o) for z, o in zip(z_axes, origins)], axis=-1)
    return J, pt


# --------------------------------------------------------------------------- trajectories
def random_moves(n_runs, rng, robot):
    """Quintic point-to-point joint moves: 2 moves per run, rest-to-rest."""
    steps = int(T_RUN / DT)
    t = np.arange(steps) * DT
    lo = np.deg2rad([-150, -60, -60, -150, -100, -180])
    hi = np.deg2rad([150, 90, 90, 150, 100, 180])
    W = lo + rng.random((n_runs, 3, 6)) * (hi - lo)
    W[:, 1] = W[:, 0] + np.clip(W[:, 1] - W[:, 0], -1.2, 1.2)
    W[:, 2] = W[:, 1] + np.clip(W[:, 2] - W[:, 1], -1.2, 1.2)
    seg = T_RUN / 2
    q, qd, qdd = (np.empty((steps, n_runs, 6)) for _ in range(3))
    for k in range(2):
        m = (t >= k * seg) & (t < (k + 1) * seg) if k == 0 else t >= k * seg
        s = np.clip((t[m] - k * seg) / seg, 0, 1)[:, None, None]
        b = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        db = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / seg
        ddb = (60 * s - 180 * s ** 2 + 120 * s ** 3) / seg ** 2
        D = (W[:, k + 1] - W[:, k])[None]
        q[m], qd[m], qdd[m] = W[:, k][None] + b * D, db * D, ddb * D
    return q, qd, qdd


@dataclass
class FleetRun:
    q: np.ndarray            # (steps, N, 6) measured
    qd: np.ndarray
    tau: np.ndarray          # commanded torque
    qdd_des: np.ndarray
    q_des: np.ndarray
    fault: np.ndarray        # (N,) fault label ("none" or one of FAULTS)
    onset: np.ndarray        # (N,) onset time (s), nan for none
    joint: np.ndarray        # (N,) affected joint for wear/actuator, -1 otherwise


def simulate_fleet(n_runs, faults=None, seed=0, dyn_nominal: ArmDynamics | None = None):
    """faults: list of length n_runs with 'none' or a fault name."""
    rng = np.random.default_rng(seed)
    nom = dyn_nominal or ArmDynamics()
    plant = ArmDynamics(payload_kg=nom.payload_kg)
    faults = np.array(faults if faults is not None else ["none"] * n_runs)
    N, steps = n_runs, int(T_RUN / DT)
    q_d, qd_d, qdd_d = random_moves(N, rng, nom.robot)
    visc = plant.viscous * rng.uniform(0.7, 1.3, (N, 6))
    coul = plant.coulomb * rng.uniform(0.7, 1.3, (N, 6))
    payload_dm = rng.uniform(-0.5, 0.5, N)                        # normal payload variation
    onset = np.where(faults == "none", np.nan, rng.uniform(1.5, 4.5, N))
    joint = np.where(np.isin(faults, ["wear", "actuator"]), rng.integers(0, 6, N), -1)
    f_dir = rng.normal(size=(N, 3))
    f_dir /= np.linalg.norm(f_dir, axis=1, keepdims=True)
    f_mag, f_dur = rng.uniform(40, 150, N), rng.uniform(0.05, 0.3, N)
    extra_mass = rng.uniform(1.5, 4.0, N)
    act_gain = rng.uniform(0.6, 0.8, N)

    q, qd = q_d[0].copy(), np.zeros((N, 6))
    Q, QD, TAU = (np.empty((steps, N, 6)) for _ in range(3))
    idx = np.arange(N)
    for i in range(steps):
        t = i * DT
        qm = q + rng.normal(0, 1e-5, q.shape)                     # encoder noise
        qdm = qd + rng.normal(0, 2e-3, qd.shape)
        # computed-torque controller on the nominal rigid-body model (no friction model)
        a_cmd = qdd_d[i] + KD * (qd_d[i] - qdm) + KP * (q_d[i] - qm)
        M_hat = nom.mass_matrix(qm)
        tau = np.einsum("nij,nj->ni", M_hat, a_cmd) + nom.bias(qm, qdm, friction=False)
        tau_meas = tau + rng.normal(0, 0.2, tau.shape)             # current-sensor noise (measurement only)
        active = ~np.isnan(onset) & (t >= onset)
        tau_motor = tau.copy()
        sel = active & (faults == "actuator")
        tau_motor[idx[sel], joint[sel]] *= act_gain[sel]
        # external wrench on the tool: payload variation (+ fault payload) and collisions
        J, _ = flange_point_jacobian(plant, q)
        dm = payload_dm + np.where(active & (faults == "payload"), extra_mass, 0.0)
        F = -(dm * G)[:, None] * np.array([0, 0, 1.0])
        col = active & (faults == "collision") & (t < onset + f_dur)
        if col.any():
            ph = np.clip((t - onset[col]) / f_dur[col], 0, 1)
            F[col] += (f_mag[col] * np.sin(np.pi * ph))[:, None] * f_dir[col]
        tau_ext = np.einsum("nij,ni->nj", J, F)
        # plant friction (unit-specific, plus wear fault), integrated linearly-implicitly for stability
        v_, c_ = visc.copy(), coul.copy()
        sel = active & (faults == "wear")
        v_[idx[sel], joint[sel]] *= 3.0
        c_[idx[sel], joint[sel]] *= 2.0
        vs = 0.02
        fric = v_ * qd + c_ * np.tanh(qd / vs)
        dfric = v_ + c_ / vs / np.cosh(qd / vs) ** 2
        M = plant.mass_matrix(q)
        rhs = tau_motor + tau_ext - plant.bias(q, qd, friction=False) - fric
        A = M + DT * dfric[:, :, None] * np.eye(6)
        dv = np.linalg.solve(A, (DT * rhs)[..., None])[..., 0]
        Q[i], QD[i], TAU[i] = qm, qdm, tau_meas
        qd = qd + dv
        q = q + DT * qd
    return FleetRun(Q, QD, TAU, qdd_d, q_d, faults, onset, joint)


# --------------------------------------------------------------------------- residuals
def naive_residual(run: FleetRun, dyn: ArmDynamics):
    steps, N, _ = run.q.shape
    q, qd, qdd = (x.reshape(-1, 6) for x in (run.q, run.qd, run.qdd_des))
    tau_id = dyn.rnea(q, qd, qdd, friction=False).reshape(steps, N, 6)
    return run.tau - tau_id


def momentum_observer(run: FleetRun, dyn: ArmDynamics, k=K_OBS):
    """r_dot = K (p_dot_measured_model - r): estimates tau_ext + unmodelled torques without qdd."""
    steps, N, _ = run.q.shape
    r = np.zeros((N, 6))
    R = np.empty((steps, N, 6))
    M_prev = dyn.mass_matrix(run.q[0])
    p0 = np.einsum("nij,nj->ni", M_prev, run.qd[0])
    integ = np.zeros((N, 6))
    for i in range(steps):
        q, qd = run.q[i], run.qd[i]
        M = dyn.mass_matrix(q) if i else M_prev
        Mdot_qd = np.einsum("nij,nj->ni", (M - M_prev) / DT, qd) if i else np.zeros((N, 6))
        b = dyn.bias(q, qd, friction=False)
        integ += (run.tau[i] - b + Mdot_qd + r) * DT
        p = np.einsum("nij,nj->ni", M, qd)
        r = k * (p - p0 - integ)
        R[i], M_prev = r, M
    return R


def residual_features(run: FleetRun):
    """Inputs for the learned normal-behaviour model: state + desired motion (no torque)."""
    return np.concatenate([run.q, run.qd, np.tanh(run.qd / 0.02), run.qdd_des], axis=-1)


def observer_filter(x, k=K_OBS):
    """The observer's own first-order dynamics (bandwidth k), applied along time (axis 0)."""
    y = np.zeros_like(x)
    a = k * DT
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


class LearnedResidual:
    """Gradient-boosted model of normal unmodelled torque, learned from fault-free runs.

    The observer residual r is a first-order low-pass (bandwidth K) of the unmodelled torque.
    So the model learns the *de-filtered* target r + r_dot / K from instantaneous state, and at
    detection time its prediction is passed through the same filter. That matches the observer's
    lag at velocity reversals, where friction steps, instead of leaving error spikes there.
    """

    def fit(self, runs_feats, runs_res, max_points=200_000, seed=0):
        from sklearn.ensemble import HistGradientBoostingRegressor
        r_dot = np.gradient(runs_res, DT, axis=0)
        target = runs_res + r_dot / K_OBS
        X = runs_feats.reshape(-1, runs_feats.shape[-1])
        Y = target.reshape(-1, 6)
        idx = np.random.default_rng(seed).choice(len(X), min(max_points, len(X)), replace=False)
        self.models = [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=seed)
                       .fit(X[idx], Y[idx, j]) for j in range(6)]
        return self

    def predict(self, feats, filtered=True):
        """feats: (steps, N, F). Returns predicted observer residual (steps, N, 6)."""
        X = feats.reshape(-1, feats.shape[-1])
        raw = np.stack([m.predict(X) for m in self.models], -1).reshape(feats.shape[:-1] + (6,))
        return observer_filter(raw) if filtered else raw


# --------------------------------------------------------------------------- detection
def score_series(res, scale, warmup_s=0.2):
    """Normalised max-over-joints residual, with a short warm-up masked out."""
    s = np.abs(res / scale).max(-1)
    s[: int(warmup_s / DT)] = 0.0
    return s


def debounce(s, k=10):
    """Running minimum over k samples: an alarm needs k consecutive samples above threshold."""
    out = s.copy()
    for j in range(1, k):
        out[k - 1:] = np.minimum(out[k - 1:], s[k - 1 - j: len(s) - j])
    out[: k - 1] = 0
    return out


def detect(score, threshold):
    """Run-level alarm and alarm time index (or -1)."""
    over = score > threshold
    hit = over.any(0)
    first = np.where(hit, over.argmax(0), -1)
    return hit, first


def diagnosis_features(run: FleetRun, res, dyn: ArmDynamics, t_idx, window_s=0.5):
    """Signature of the residual after the alarm: per-joint level, spread and correlation with
    velocity (friction-like), commanded torque (actuator-like) and gravity load (payload-like)."""
    w = int(window_s / DT)
    feats = []
    for n, i0 in enumerate(t_idx):
        i0 = int(min(max(i0, 0), res.shape[0] - w))
        sl = slice(i0, i0 + w)
        r, qd, tau = res[sl, n], run.qd[sl, n], run.tau[sl, n]
        g = dyn.gravity(run.q[sl, n][:: 25])
        r_sub = r[:: 25]

        def corr(a, b):
            a, b = a - a.mean(0), b - b.mean(0)
            return (a * b).sum(0) / (np.sqrt((a * a).sum(0) * (b * b).sum(0)) + 1e-9)

        peak = np.abs(r).max(0)
        feats.append(np.concatenate([r.mean(0), r.std(0), peak / (peak.sum() + 1e-9),
                                     corr(r, qd), corr(r, tau), corr(r_sub, g),
                                     [np.abs(r).max() / (np.abs(r[-w // 5:]).mean() + 1e-9)]]))
    return np.array(feats)
