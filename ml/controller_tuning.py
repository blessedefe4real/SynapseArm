"""Phase 2: learned, robust controller tuning for a mass-spring-damper axis (m=7, b=1, k=9).

The classical partitioned (computed-torque) controller f = m f' + b x' + k x is perfect when the model is exact.
Real plants aren't, so we tune gains for *robust* performance across a distribution of
scenarios:

  * true m, b, k each off by up to +/-30% from the nominal model
  * an unknown step load disturbance (up to +/-3 N) at t = 10 s
  * actuator saturation |f| <= 30 N, plus sensor noise on position and velocity

Each candidate gain set is scored on a whole batch of scenarios at once (vectorised), and a
Gaussian-process Bayesian optimiser searches the gains. Common random numbers (same scenarios
for every candidate) make the objective deterministic, which suits the GP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotics.control import partitioned_pd_gains, routh_stable

M, B, K = 7.0, 1.0, 9.0          # nominal plant model
U_MAX = 30.0                     # actuator limit (N)
T_END, DT, T_DIST = 20.0, 2e-3, 10.0
X_REF = 1.0


@dataclass
class Scenarios:
    m: np.ndarray
    b: np.ndarray
    k: np.ndarray
    d: np.ndarray                # step disturbance magnitude (N)
    noise_x: np.ndarray          # (steps, n)
    noise_v: np.ndarray

    @property
    def n(self):
        return len(self.m)


def make_scenarios(n=24, seed=0, model_err=0.3, d_max=3.0, sigma_x=1e-3, sigma_v=5e-3) -> Scenarios:
    rng = np.random.default_rng(seed)
    steps = int(T_END / DT)
    u = lambda: rng.uniform(1 - model_err, 1 + model_err, n)  # noqa: E731
    return Scenarios(M * u(), B * u(), K * u(), rng.uniform(-d_max, d_max, n),
                     rng.normal(0, sigma_x, (steps, n)), rng.normal(0, sigma_v, (steps, n)))


def nominal_scenario() -> Scenarios:
    steps = int(T_END / DT)
    z = np.zeros((steps, 1))
    return Scenarios(np.array([M]), np.array([B]), np.array([K]), np.array([0.0]), z, z)


def simulate(kind: str, gains, sc: Scenarios):
    """Vectorised closed-loop simulation over all scenarios.

    kind = "partitioned": f = M (kp e + kd e' + ki ∫e) + B x' + K x   (model-based, computed-torque)
    kind = "pid":         f = kp e + ki ∫e + kd e'                    (model-free)
    Returns t, x (steps, n), u (steps, n).
    """
    kp, kd, ki = gains
    steps, n = int(T_END / DT), sc.n
    t = np.arange(steps) * DT
    x, v, integ = np.zeros(n), np.zeros(n), np.zeros(n)
    X, U = np.empty((steps, n)), np.empty((steps, n))
    i_dist = int(T_DIST / DT)
    for i in range(steps):
        xm, vm = x + sc.noise_x[i], v + sc.noise_v[i]
        e = X_REF - xm
        if kind == "partitioned":
            f = M * (kp * e - kd * vm + ki * integ) + B * vm + K * xm
        else:
            f = kp * e - kd * vm + ki * integ
        f_sat = np.clip(f, -U_MAX, U_MAX)
        # conditional-integration anti-windup: freeze the integrator while saturated
        integ += np.where(f == f_sat, e, 0.0) * DT
        d = sc.d if i >= i_dist else 0.0
        a = (f_sat + d - sc.b * v - sc.k * x) / sc.m
        v = v + a * DT
        x = x + v * DT
        X[i], U[i] = x, f_sat
    return t, X, U


def metrics(t, X, U):
    """Per-scenario performance numbers."""
    e = X_REF - X
    iae = np.abs(e).sum(0) * DT
    tv = np.abs(np.diff(U, axis=0)).sum(0)            # control total variation (chatter / effort)
    pre = t < T_DIST
    overshoot = np.maximum(X[pre].max(0) - X_REF, 0) / X_REF * 100
    # 2% settling time for the set-point step: first time after which |e| stays inside the band
    outside = np.abs(e[pre]) > 0.02 * X_REF
    n_pre = outside.shape[0]
    last_out = n_pre - 1 - np.argmax(outside[::-1], axis=0)   # last index outside the band
    settle = t[np.minimum(last_out + 1, n_pre - 1)]
    settle = np.where(outside[-1], np.inf, settle)            # never settled before t_dist
    ss_err = np.abs(e[-int(1.0 / DT):]).mean(0)          # residual error after the disturbance
    return {"iae": iae, "tv": tv, "overshoot_pct": overshoot, "settle_s": settle, "ss_err": ss_err}


TV_WEIGHT = 0.002        # effort / sensor-noise chatter
OVERSHOOT_WEIGHT = 0.05  # per % overshoot: 20% overshoot costs as much as 1.0 IAE


def _per_scenario_cost(m):
    return m["iae"] + TV_WEIGHT * m["tv"] + OVERSHOOT_WEIGHT * m["overshoot_pct"]


def robust_cost(kind, gains, sc: Scenarios) -> float:
    """Scalar objective: mean + 0.5 std over scenarios of IAE + chatter + overshoot penalties."""
    j = _per_scenario_cost(metrics(*simulate(kind, gains, sc)))
    if not np.all(np.isfinite(j)):
        return 1e3
    return float(np.mean(j) + 0.5 * np.std(j))


# Search boxes (kp, kd, ki), searched in log10 space.
BOUNDS = {
    # servo gains of f' (units of acceleration per unit error)
    "partitioned": [(0.05, 50.0), (0.1, 30.0), (0.01, 30.0)],
    # raw force gains
    "pid": [(0.5, 300.0), (0.5, 200.0), (0.05, 200.0)],
}


def classical_baseline_gains():
    """Textbook partitioned design: zeta = 1, wn = 0.5 sqrt(k/m), no integral action."""
    g = partitioned_pd_gains(M, B, K)
    return np.array([g["kp"], g["kd"], 0.0])


def closed_loop_poly(kind, gains, m, b, k):
    """Characteristic polynomial of the linear closed loop (saturation ignored) on a true plant m, b, k."""
    kp, kd, ki = gains
    if kind == "partitioned":   # f = M(kp e - kd v + ki∫e) + B v + K x  -> spring/damper cancel imperfectly
        c = [m, M * kd + b - B, M * kp + k - K, M * ki]
    else:
        c = [m, b + kd, k + kp, ki]
    return c if ki > 0 else c[:3]


def linear_stable(kind, gains, sc: Scenarios) -> np.ndarray:
    """Routh-Hurwitz test for every scenario's true plant."""
    return np.array([routh_stable(closed_loop_poly(kind, gains, m, b, k))
                     for m, b, k in zip(sc.m, sc.b, sc.k)])


def summarise(kind, gains, sc: Scenarios) -> dict:
    m = metrics(*simulate(kind, gains, sc))
    j = _per_scenario_cost(m)
    finite_settle = m["settle_s"][np.isfinite(m["settle_s"])]
    return {
        "gains_kp_kd_ki": [float(g) for g in gains],
        "robust_cost": float(np.mean(j) + 0.5 * np.std(j)),
        "mean_iae": float(m["iae"].mean()),
        "worst_iae": float(m["iae"].max()),
        "unstable_frac": float(1 - np.mean(linear_stable(kind, gains, sc))),
        "mean_overshoot_pct": float(m["overshoot_pct"].mean()),
        "median_settle_s": float(np.median(m["settle_s"])),
        "settled_frac": float(len(finite_settle) / len(m["settle_s"])),
        "mean_final_err": float(m["ss_err"].mean()),
        "mean_control_tv": float(m["tv"].mean()),
    }
