"""Classical control: Routh-Hurwitz stability and control-law partitioning (computed-torque style).

These are the classical baselines that the ML controllers in the roadmap will be compared against.
"""
from __future__ import annotations

import numpy as np


def routh_stable(coeffs) -> bool:
    """True if all roots of the polynomial (highest power first) lie in the open left half-plane."""
    roots = np.roots(coeffs)
    return bool(np.all(roots.real < 0))


def stable_gain_range(char_poly, k_values) -> np.ndarray:
    """Evaluate stability over a grid of gains. char_poly(K) -> coefficient list."""
    return np.array([routh_stable(char_poly(k)) for k in k_values])


def type1_plant_char_poly(K):
    """Plant 1/(s(s+2)(s+3)) with compensator K(s+1): s(s+2)(s+3) + K(s+1) = s^3 + 5s^2 + (K+6)s + K  (stable for K > 0)."""
    return [1.0, 5.0, K + 6.0, K]


def partitioned_pd_gains(m: float, b: float, k: float, zeta: float = 1.0, wn_ratio: float = 0.5):
    """Control-law partitioning f = alpha f' + beta for m x'' + b x' + k x = f.

    alpha = m, beta = b x' + k x, servo f' = x''_d + kd e' + kp e with
    kp = wn^2, kd = 2 zeta wn, wn = wn_ratio * sqrt(k/m).
    """
    w_res = np.sqrt(k / m)
    wn = wn_ratio * w_res
    return {"alpha": m, "beta": (b, k), "w_res": w_res, "wn": wn, "kp": wn ** 2, "kd": 2 * zeta * wn}


def simulate_msd(m, b, k, controller, x0=0.0, v0=0.0, x_ref=1.0, t_end=30.0, dt=1e-3,
                 disturbance=None, model_error=(1.0, 1.0, 1.0)):
    """Simulate m x'' + b x' + k x = f + d(t) with a controller f = controller(t, x, v, x_ref).

    model_error scales (m, b, k) of the *true* plant relative to the nominal model, so we can
    test robustness of model-based vs learned controllers.
    """
    mt, bt, kt = m * model_error[0], b * model_error[1], k * model_error[2]
    n = int(t_end / dt)
    t = np.arange(n) * dt
    x, v = np.empty(n), np.empty(n)
    u = np.empty(n)
    x[0], v[0] = x0, v0
    for i in range(n - 1):
        f = controller(t[i], x[i], v[i], x_ref)
        d = disturbance(t[i]) if disturbance else 0.0
        a = (f + d - bt * v[i] - kt * x[i]) / mt
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i + 1] * dt
        u[i] = f
    u[-1] = u[-2]
    return t, x, v, u


def make_partitioned_controller(m, b, k, kp, kd):
    def ctrl(t, x, v, x_ref):
        e, edot = x_ref - x, -v
        f_prime = kp * e + kd * edot  # x''_d = 0 for a set-point
        return m * f_prime + b * v + k * x
    return ctrl
