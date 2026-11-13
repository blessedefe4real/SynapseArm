"""Kinematics of a 6-DOF industrial arm (ABB IRB-style geometry: offset shoulder, spherical wrist).

Standard Denavit-Hartenberg convention:
    T_i = Rot_z(theta_i) . Trans_z(d_i) . Trans_x(a_i) . Rot_x(alpha_i)

The parameters give the closed-form wrist-centre position (derived by hand, verified in tests):
    Px = c1 (150 + 760 c2 + 140 c23 + 795 s23)
    Py = s1 (150 + 760 c2 + 140 c23 + 795 s23)
    Pz = 505 + 760 s2 + 140 s23 - 795 c23
All lengths are in millimetres, angles in radians.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PI = np.pi


@dataclass
class DHRobot:
    # a, alpha, d per link (theta is the joint variable)
    a: np.ndarray = field(default_factory=lambda: np.array([150.0, 760.0, 140.0, 0.0, 0.0, 0.0]))
    alpha: np.ndarray = field(default_factory=lambda: np.array([PI / 2, 0.0, PI / 2, -PI / 2, PI / 2, 0.0]))
    d: np.ndarray = field(default_factory=lambda: np.array([505.0, 0.0, 0.0, 795.0, 0.0, 0.0]))
    theta_offset: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # Joint limits (rad). Typical ABB-style limits.
    q_min: np.ndarray = field(default_factory=lambda: np.deg2rad([-180, -100, -165, -270, -130, -400]))
    q_max: np.ndarray = field(default_factory=lambda: np.deg2rad([180, 155, 255, 270, 130, 400]))

    @property
    def n(self) -> int:
        return len(self.a)

    # ------------------------------------------------------------------ FK
    @staticmethod
    def dh_matrix(theta, d, a, alpha) -> np.ndarray:
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def link_transforms(self, q) -> list[np.ndarray]:
        """Cumulative transforms 0T1, 0T2, ..., 0T6 (index 0 is identity / base)."""
        q = np.asarray(q, dtype=float)
        T = np.eye(4)
        frames = [T]
        for i in range(self.n):
            T = T @ self.dh_matrix(q[i] + self.theta_offset[i], self.d[i], self.a[i], self.alpha[i])
            frames.append(T)
        return frames

    def fk(self, q) -> np.ndarray:
        """End-effector pose 0T6 as a 4x4 homogeneous matrix."""
        return self.link_transforms(q)[-1]

    def fk_batch(self, Q: np.ndarray) -> np.ndarray:
        """Vectorised FK for an (N, 6) array of joint vectors -> (N, 4, 4)."""
        Q = np.atleast_2d(Q)
        N = Q.shape[0]
        T = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
        for i in range(self.n):
            th = Q[:, i] + self.theta_offset[i]
            ct, st = np.cos(th), np.sin(th)
            ca, sa = np.cos(self.alpha[i]), np.sin(self.alpha[i])
            A = np.zeros((N, 4, 4))
            A[:, 0, 0], A[:, 0, 1], A[:, 0, 2], A[:, 0, 3] = ct, -st * ca, st * sa, self.a[i] * ct
            A[:, 1, 0], A[:, 1, 1], A[:, 1, 2], A[:, 1, 3] = st, ct * ca, -ct * sa, self.a[i] * st
            A[:, 2, 1], A[:, 2, 2], A[:, 2, 3] = sa, ca, self.d[i]
            A[:, 3, 3] = 1.0
            T = T @ A
        return T

    # ------------------------------------------------------------ Jacobian
    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian (6x6): rows 0-2 linear velocity, rows 3-5 angular velocity.

        J_v,i = z_{i-1} x (o_6 - o_{i-1}),   J_w,i = z_{i-1}   (all joints revolute)
        """
        frames = self.link_transforms(q)
        o_n = frames[-1][:3, 3]
        J = np.zeros((6, self.n))
        for i in range(self.n):
            z = frames[i][:3, 2]
            o = frames[i][:3, 3]
            J[:3, i] = np.cross(z, o_n - o)
            J[3:, i] = z
        return J

    def manipulability(self, q) -> float:
        """Yoshikawa manipulability index sqrt(det(J J^T)); 0 at singularities."""
        J = self.jacobian(q)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    # ------------------------------------------------------------------ IK
    @staticmethod
    def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        """6-vector [position error (mm); orientation error (rad, axis-angle-ish)]."""
        ep = T_target[:3, 3] - T_current[:3, 3]
        Rc, Rt = T_current[:3, :3], T_target[:3, :3]
        eo = 0.5 * (np.cross(Rc[:, 0], Rt[:, 0]) + np.cross(Rc[:, 1], Rt[:, 1]) + np.cross(Rc[:, 2], Rt[:, 2]))
        return np.concatenate([ep, eo])

    def ik_numeric(self, T_target, q0=None, max_iter=200, tol_pos=1e-3, tol_rot=1e-5,
                   damping=1.0, orient_weight=500.0, clamp=True):
        """Damped-least-squares (Levenberg-Marquardt) numerical IK.

        Returns (q, success, iterations). orient_weight scales rotation error (rad) to be
        comparable with position error (mm).
        """
        q = np.zeros(self.n) if q0 is None else np.array(q0, dtype=float)
        W = np.diag([1, 1, 1, orient_weight, orient_weight, orient_weight])
        for it in range(1, max_iter + 1):
            T = self.fk(q)
            e = self.pose_error(T, T_target)
            if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
                return q, True, it
            J = W @ self.jacobian(q)
            ew = W @ e
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), ew)
            q = q + dq
            if clamp:
                q = np.clip(q, self.q_min, self.q_max)
        T = self.fk(q)
        e = self.pose_error(T, T_target)
        return q, bool(np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot), max_iter

    def ik_position_analytic(self, p, elbow: str = "up"):
        """Closed-form solution for joints 1-3 that place the wrist centre at p (d6 = 0).

        Geometric approach: theta1 = atan2(y, x), then the cosine rule in the arm plane
        for theta2/theta3.
        """
        x, y, z = p
        a1, a2, a3, d1, d4 = self.a[0], self.a[1], self.a[2], self.d[0], self.d[3]
        q1 = np.arctan2(y, x)
        r = np.hypot(x, y) - a1
        s = z - d1
        L3 = np.hypot(a3, d4)                 # 807.23 mm, effective forearm length
        phi = np.arctan2(d4, a3)              # fixed offset angle of the forearm
        D = (r ** 2 + s ** 2 - a2 ** 2 - L3 ** 2) / (2 * a2 * L3)
        if abs(D) > 1:
            raise ValueError("Target outside reachable workspace")
        gamma = np.arctan2(np.sqrt(1 - D ** 2) * (1 if elbow == "up" else -1), D)
        q2 = np.arctan2(s, r) - np.arctan2(L3 * np.sin(gamma), a2 + L3 * np.cos(gamma))
        q3 = gamma + phi
        return np.array([q1, q2, q3])

    # ------------------------------------------------------------- helpers
    def random_q(self, n: int, rng=None, q_min=None, q_max=None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        lo = self.q_min if q_min is None else q_min
        hi = self.q_max if q_max is None else q_max
        return rng.uniform(lo, hi, size=(n, self.n))


def wrap_to_pi(q):
    return (np.asarray(q) + PI) % (2 * PI) - PI
