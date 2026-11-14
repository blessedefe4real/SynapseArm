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
class Units:
    """Unit-to-unit physical variation of each arm in the fleet (kept so a unit can be re-run)."""
    visc: np.ndarray         # (N, 6) viscous friction
    coul: np.ndarray         # (N, 6) Coulomb friction
    strib: np.ndarray        # (N, 6) Stribeck (static-friction) excess, not in any model's basis
    payload_dm: np.ndarray   # (N,) payload deviation from nominal (kg)

    @classmethod
    def sample(cls, n, rng, dyn: ArmDynamics):
        return cls(dyn.viscous * rng.uniform(0.7, 1.3, (n, 6)), dyn.coulomb * rng.uniform(0.7, 1.3, (n, 6)),
                   dyn.coulomb * rng.uniform(0.0, 0.6, (n, 6)), rng.uniform(-0.5, 0.5, n))

    def __getitem__(self, sl):
        return Units(self.visc[sl], self.coul[sl], self.strib[sl], self.payload_dm[sl])

    @classmethod
    def concat(cls, parts):
        return cls(*(np.concatenate([getattr(u, f) for u in parts]) for f in ("visc", "coul", "strib", "payload_dm")))


V_STRIBECK = 0.1   # rad/s


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
    units: Units | None = None
    qd_true: np.ndarray | None = None   # ground-truth velocity (simulation only; for tuning/tests)


def simulate_fleet(n_runs, faults=None, seed=0, dyn_nominal: ArmDynamics | None = None, units: Units | None = None):
    """faults: list of length n_runs with 'none' or a fault name. Pass `units` to re-run the same
    physical arms (e.g. a commissioning run) on new trajectories."""
    rng = np.random.default_rng(seed)
    nom = dyn_nominal or ArmDynamics()
    plant = ArmDynamics(payload_kg=nom.payload_kg)
    faults = np.array(faults if faults is not None else ["none"] * n_runs)
    N, steps = n_runs, int(T_RUN / DT)
    q_d, qd_d, qdd_d = random_moves(N, rng, nom.robot)
    if units is None:
        units = Units.sample(N, np.random.default_rng(seed + 10_000), plant)
    visc, coul, strib, payload_dm = units.visc, units.coul, units.strib, units.payload_dm
    onset = np.where(faults == "none", np.nan, rng.uniform(1.5, 4.5, N))
    joint = np.where(np.isin(faults, ["wear", "actuator"]), rng.integers(0, 6, N), -1)
    f_dir = rng.normal(size=(N, 3))
    f_dir /= np.linalg.norm(f_dir, axis=1, keepdims=True)
    f_mag, f_dur = rng.uniform(40, 150, N), rng.uniform(0.05, 0.3, N)
    extra_mass = rng.uniform(1.5, 4.0, N)
    act_gain = rng.uniform(0.6, 0.8, N)

    q, qd = q_d[0].copy(), np.zeros((N, 6))
    Q, QD, TAU, QDT = (np.empty((steps, N, 6)) for _ in range(4))
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
        st = strib * np.exp(-(qd / V_STRIBECK) ** 2)               # Stribeck: extra friction near rest
        th = np.tanh(qd / vs)
        fric = v_ * qd + (c_ + st) * th
        dfric = v_ + (c_ + st) / vs / np.cosh(qd / vs) ** 2 - 2 * qd / V_STRIBECK ** 2 * st * th
        M = plant.mass_matrix(q)
        rhs = tau_motor + tau_ext - plant.bias(q, qd, friction=False) - fric
        A = M + DT * dfric[:, :, None] * np.eye(6)
        dv = np.linalg.solve(A, (DT * rhs)[..., None])[..., 0]
        Q[i], QD[i], TAU[i], QDT[i] = qm, qdm, tau_meas, qd
        qd = qd + dv
        q = q + DT * qd
    return FleetRun(Q, QD, TAU, qdd_d, q_d, faults, onset, joint, units, QDT)


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


# --------------------------------------------------------------------------- velocity estimation
SIGMA_Q, SIGMA_V = 1e-5, 2e-3          # encoder position / velocity noise (rad, rad/s), as simulated


def kalman_gain(sigma_a, sigma_q=SIGMA_Q, sigma_v=SIGMA_V, iters=5000):
    """Steady-state gain of the per-joint filter, state [q, qd], measurements [q, qd].

    Process model: constant acceleration over one sample, driven by the planned acceleration
    plus white acceleration noise sigma_a (which covers tracking error, friction, faults)."""
    F = np.array([[1.0, DT], [0.0, 1.0]])
    Gm = np.array([[DT ** 2 / 2], [DT]])
    Qn = sigma_a ** 2 * Gm @ Gm.T
    R = np.diag([sigma_q ** 2, sigma_v ** 2])
    P = np.eye(2)
    for _ in range(iters):
        P = F @ P @ F.T + Qn
        Kg = P @ np.linalg.inv(P + R)
        P = (np.eye(2) - Kg) @ P
    return Kg


# per-joint process noise, tuned for lowest velocity RMSE against ground truth on a separate
# fault-free tuning fleet (scripts/fault_detection.py never sees it): big joints 0.1, wrist 0.3
SIGMA_A = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])


def kalman_velocity(run: FleetRun, sigma_a=SIGMA_A):
    """Filtered copy of a run: q and qd replaced by the steady-state Kalman estimate.

    The input is the *planned* acceleration (known to the controller, independent of any
    fault), so a fault shows up through the measurements and is not explained away by the
    prediction."""
    from dataclasses import replace
    sig = np.broadcast_to(np.asarray(sigma_a, float), (6,))
    K = np.stack([kalman_gain(sa) for sa in sig])               # (6, 2, 2)
    k00, k01, k10, k11 = K[:, 0, 0], K[:, 0, 1], K[:, 1, 0], K[:, 1, 1]
    steps = run.q.shape[0]
    q_hat, v_hat = run.q[0].copy(), run.qd[0].copy()
    Qf, Vf = np.empty_like(run.q), np.empty_like(run.qd)
    for i in range(steps):
        if i:
            a = run.qdd_des[i - 1]
            q_hat = q_hat + DT * v_hat + 0.5 * DT ** 2 * a
            v_hat = v_hat + DT * a
        eq, ev = run.q[i] - q_hat, run.qd[i] - v_hat
        q_hat = q_hat + k00 * eq + k01 * ev
        v_hat = v_hat + k10 * eq + k11 * ev
        Qf[i], Vf[i] = q_hat, v_hat
    return replace(run, q=Qf, qd=Vf)


# --------------------------------------------------------------------------- per-unit calibration
def payload_gravity_basis(run: FleetRun, dyn: ArmDynamics, chunk=200):
    """Joint torque per kg of extra payload at the tool, J^T (0, 0, -g), along the run: (steps, N, 6)."""
    steps, N, _ = run.q.shape
    out = np.empty((steps, N, 6))
    for i0 in range(0, steps, chunk):
        q = run.q[i0:i0 + chunk].reshape(-1, 6)
        J, _ = flange_point_jacobian(dyn, q)
        out[i0:i0 + chunk] = np.einsum("nij,i->nj", J, np.array([0, 0, -G])).reshape(-1, N, 6)
    return out


class UnitCalibration:
    """Identify each arm's own deviation from the fleet model from one fault-free commissioning run.

    Per arm and joint, a ridge fit of the remaining residual on a standard identification basis:
    [viscous q_dot_j, Coulomb tanh(q_dot_j / v_s), payload J^T g, offset]. Through the observer's
    filter when the residual is an observer residual. Four numbers per joint, frozen afterwards;
    the Stribeck part of the plant is deliberately not in the basis.
    """

    def __init__(self, dyn: ArmDynamics, filtered=True, ridge=1e-3):
        self.dyn, self.filtered, self.ridge = dyn, filtered, ridge

    def basis(self, run: FleetRun):
        g = payload_gravity_basis(run, self.dyn)
        B = np.stack([run.qd, np.tanh(run.qd / 0.02), g, np.ones_like(run.qd)], axis=-1)   # (steps, N, 6, 4)
        return observer_filter(B) if self.filtered else B

    def fit(self, commissioning: FleetRun, residual, warmup=100):
        B = self.basis(commissioning)[warmup:]
        r = residual[warmup:]
        scale = np.sqrt((B ** 2).mean(0)) + 1e-9                                            # (N, 6, 4)
        Bn = B / scale
        A = np.einsum("tnjk,tnjl->njkl", Bn, Bn) + self.ridge * len(B) * np.eye(4)
        b = np.einsum("tnjk,tnj->njk", Bn, r)
        self.w = np.linalg.solve(A, b[..., None])[..., 0] / scale
        return self

    def apply(self, run: FleetRun, residual):
        return residual - np.einsum("tnjk,njk->tnj", self.basis(run), self.w)


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
