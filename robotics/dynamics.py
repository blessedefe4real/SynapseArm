"""Rigid-body dynamics of the 6-DOF arm: batched recursive Newton-Euler (RNEA).

    tau = M(q) qdd + C(q, qd) qd + g(q)

Everything is vectorised over a leading batch axis, so hundreds of robots (or the 6 columns of
M plus the bias term) go through one call. Units are SI inside (m, kg, s); the kinematic model
in mm is converted once.

Inertial model: each body is a slender rod along its D-H link segment (from the previous frame
origin to its own), plus a solid-cylinder radius term, with IRB-class masses. A payload hangs
on the flange. These are plausible values, not a datasheet. The algorithms don't depend on them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robotics.kinematics import DHRobot

G = 9.81


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


@dataclass
class ArmDynamics:
    robot: DHRobot = field(default_factory=DHRobot)
    mass: np.ndarray = field(default_factory=lambda: np.array([35.0, 22.0, 14.0, 8.0, 3.0, 0.5]))
    radius: np.ndarray = field(default_factory=lambda: np.array([0.12, 0.08, 0.07, 0.06, 0.05, 0.04]))
    payload_kg: float = 2.0
    payload_offset_m: tuple = (0.0, 0.0, 0.10)        # in flange frame
    viscous: np.ndarray = field(default_factory=lambda: np.array([8.0, 8.0, 5.0, 1.5, 1.0, 0.5]))
    coulomb: np.ndarray = field(default_factory=lambda: np.array([6.0, 6.0, 4.0, 1.0, 0.8, 0.4]))

    def __post_init__(self):
        r = self.robot
        self.a = r.a / 1000.0
        self.d = r.d / 1000.0
        self.alpha = r.alpha
        self.n = r.n
        self._inertials()

    def _inertials(self):
        """COM (in own frame) and inertia about the COM, per body; payload merged into body 6."""
        self.com = np.zeros((self.n, 3))
        self.I = np.zeros((self.n, 3, 3))
        self.m = self.mass.astype(float).copy()
        for i in range(self.n):
            # previous origin expressed in frame i: Rx(-alpha) (-a, 0, -d), independent of theta
            seg = _rot_x(-self.alpha[i]) @ np.array([-self.a[i], 0.0, -self.d[i]])
            L = np.linalg.norm(seg)
            u = seg / L if L > 1e-9 else np.array([0.0, 0.0, 1.0])
            m, rr = self.mass[i], self.radius[i]
            self.com[i] = 0.5 * seg
            I_rod = m * L ** 2 / 12 * (np.eye(3) - np.outer(u, u))
            I_cyl = m * rr ** 2 / 4 * (np.eye(3) - np.outer(u, u)) + m * rr ** 2 / 2 * np.outer(u, u)
            self.I[i] = I_rod + I_cyl
        if self.payload_kg > 0:                       # parallel-axis merge of a point payload
            i = self.n - 1
            mp, cp = self.payload_kg, np.asarray(self.payload_offset_m, float)
            m0, c0 = self.m[i], self.com[i].copy()
            mt = m0 + mp
            c = (m0 * c0 + mp * cp) / mt
            shift = lambda m_, r_: m_ * (r_ @ r_ * np.eye(3) - np.outer(r_, r_))  # noqa: E731
            self.I[i] = self.I[i] + shift(m0, c0 - c) + shift(mp, cp - c)
            self.m[i], self.com[i] = mt, c

    # ----------------------------------------------------------------- kinematics
    def _link_rot_pos(self, q):
        """Per-link rotation R_i (prev->i) and p*_i (origin i from origin i-1, in frame i). q: (N, 6)."""
        N = q.shape[0]
        ct, st = np.cos(q), np.sin(q)
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        R = np.empty((N, self.n, 3, 3))
        R[..., 0, 0], R[..., 0, 1], R[..., 0, 2] = ct, -st * ca, st * sa
        R[..., 1, 0], R[..., 1, 1], R[..., 1, 2] = st, ct * ca, -ct * sa
        R[..., 2, 0], R[..., 2, 1], R[..., 2, 2] = 0.0, np.broadcast_to(sa, ct.shape), np.broadcast_to(ca, ct.shape)
        pstar = np.broadcast_to(np.stack([self.a, self.d * sa, self.d * ca], axis=-1), (N, self.n, 3))
        return R, pstar

    # ----------------------------------------------------------------- RNEA
    def rnea(self, q, qd, qdd, gravity=True, friction=False):
        """Joint torques for a batch. q, qd, qdd: (N, 6) or (6,). Returns same leading shape."""
        q, qd, qdd = (np.atleast_2d(np.asarray(x, float)) for x in (q, qd, qdd))
        N = q.shape[0]
        R, ps = self._link_rot_pos(q)
        RT = np.swapaxes(R, -1, -2)
        z = np.array([0.0, 0.0, 1.0])
        w = np.zeros((N, 3))
        dw = np.zeros((N, 3))
        a = np.broadcast_to(np.array([0.0, 0.0, G if gravity else 0.0]), (N, 3)).copy()
        F, Nm = np.empty((N, self.n, 3)), np.empty((N, self.n, 3))
        for i in range(self.n):
            Ri_T = RT[:, i]
            w_prev = w
            w = np.einsum("nij,nj->ni", Ri_T, w_prev + qd[:, i:i + 1] * z)
            dw = np.einsum("nij,nj->ni", Ri_T, dw + qdd[:, i:i + 1] * z + qd[:, i:i + 1] * np.cross(w_prev, z))
            p = ps[:, i]
            a = (np.einsum("nij,nj->ni", Ri_T, a) + np.cross(dw, p) + np.cross(w, np.cross(w, p)))
            rc = self.com[i]
            ac = a + np.cross(dw, rc) + np.cross(w, np.cross(w, rc))
            F[:, i] = self.m[i] * ac
            Iw = w @ self.I[i].T
            Nm[:, i] = dw @ self.I[i].T + np.cross(w, Iw)
        tau = np.empty((N, self.n))
        f = np.zeros((N, 3))
        nn = np.zeros((N, 3))
        for i in reversed(range(self.n)):
            if i < self.n - 1:
                Rn = R[:, i + 1]
                f_next_in_i = np.einsum("nij,nj->ni", Rn, f)
                n_next_in_i = np.einsum("nij,nj->ni", Rn, nn)
            else:
                f_next_in_i = n_next_in_i = np.zeros((N, 3))
            p, rc = ps[:, i], self.com[i]
            f_new = f_next_in_i + F[:, i]
            nn = n_next_in_i + np.cross(p, f_next_in_i) + np.cross(p + rc, F[:, i]) + Nm[:, i]
            f = f_new
            # joint axis z_{i-1} expressed in frame i
            tau[:, i] = np.einsum("nj,nj->n", nn, RT[:, i] @ z)
        if friction:
            tau = tau + self.viscous * qd + self.coulomb * np.tanh(qd / 0.02)
        return tau

    # ----------------------------------------------------------------- derived quantities
    def mass_matrix(self, q):
        """M(q) for a batch: (N, 6, 6), from 6 RNEA calls with unit accelerations, batched."""
        q = np.atleast_2d(q)
        N = q.shape[0]
        Q = np.repeat(q, self.n, axis=0)
        E = np.tile(np.eye(self.n), (N, 1))
        cols = self.rnea(Q, np.zeros_like(Q), E, gravity=False)
        return cols.reshape(N, self.n, self.n).swapaxes(1, 2)

    def bias(self, q, qd, friction=True):
        """C(q, qd) qd + g(q) (+ friction)."""
        return self.rnea(q, qd, np.zeros_like(np.atleast_2d(q)), gravity=True, friction=friction)

    def gravity(self, q):
        q = np.atleast_2d(q)
        return self.rnea(q, np.zeros_like(q), np.zeros_like(q), gravity=True)

    def forward(self, q, qd, tau, friction=True):
        """Joint accelerations qdd = M^-1 (tau - bias)."""
        M = self.mass_matrix(q)
        return np.linalg.solve(M, (np.atleast_2d(tau) - self.bias(q, qd, friction))[..., None])[..., 0]

    def com_positions(self, q):
        """World positions of each body's COM (m), single configuration."""
        frames = self.robot.link_transforms(q)
        out = []
        for i in range(self.n):
            T = frames[i + 1]
            out.append(T[:3, :3] @ self.com[i] + T[:3, 3] / 1000.0)
        return np.array(out)

    def potential_energy(self, q):
        return float(G * (self.m * self.com_positions(q)[:, 2]).sum())

    def kinetic_energy(self, q, qd):
        M = self.mass_matrix(q)[0]
        return float(0.5 * qd @ M @ qd)
