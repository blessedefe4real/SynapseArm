"""ML feature #1 - Neural-network inverse kinematics with numerical refinement.

Idea
----
A closed-form solution only exists for the wrist-centre position (joints 1-3). A full 6-DOF numerical solver works but
needs a good starting guess: from a poor seed it is slow, can converge to the wrong
configuration or fail near joint limits/singularities.

We train a multilayer perceptron on (pose -> joints) pairs generated from our own forward
kinematics, so no external data is needed. The network is fast but only ~mm accurate, so its
output is used as the *seed* for damped-least-squares IK, which then converges in a few
iterations to sub-micron accuracy. That is the "hybrid" solver.

IK is one-to-many (elbow up/down, wrist flip, front/back), and a regression network averages
between solutions if you train it on all of them. We therefore train on a single
configuration branch (elbow-down, wrist-non-flipped, front-reach), the same thing an
industrial controller does with its "confdata". Joint angles are predicted as (sin, cos)
pairs so the 2*pi wrap-around of the wrist joints does not confuse the network.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from robotics.kinematics import DHRobot

DEG = np.pi / 180

# Configuration branch used for training (radians)
BRANCH_MIN = np.array([-180, -100, -100, -180, 10, -180]) * DEG
BRANCH_MAX = np.array([180, 155, 75, 180, 125, 180]) * DEG


def sample_branch(robot: DHRobot, n: int, seed=None) -> np.ndarray:
    """Sample joint vectors from the training branch, keeping only front-reach poses."""
    rng = np.random.default_rng(seed)
    out = []
    while sum(len(o) for o in out) < n:
        Q = rng.uniform(BRANCH_MIN, BRANCH_MAX, size=(2 * n, 6))
        q2, q23 = Q[:, 1], Q[:, 1] + Q[:, 2]
        radial = robot.a[0] + robot.a[1] * np.cos(q2) + robot.a[2] * np.cos(q23) + robot.d[3] * np.sin(q23)
        out.append(Q[radial > robot.a[0] + 50])  # wrist centre in front of the shoulder
    return np.concatenate(out)[:n]


def pose_features(T: np.ndarray) -> np.ndarray:
    """(N,4,4) poses -> (N,12) features: position in metres + flattened rotation matrix."""
    T = np.asarray(T)
    if T.ndim == 2:
        T = T[None]
    return np.concatenate([T[:, :3, 3] / 1000.0, T[:, :3, :3].reshape(len(T), 9)], axis=1)


def encode_joints(Q):
    return np.concatenate([np.sin(Q), np.cos(Q)], axis=1)


def decode_joints(Y):
    return np.arctan2(Y[:, :6], Y[:, 6:])


@dataclass
class NeuralIK:
    robot: DHRobot
    model: MLPRegressor | None = None
    scaler: StandardScaler | None = None

    def fit(self, n_samples=150_000, hidden=(256, 256, 256), max_iter=200, seed=0, verbose=True):
        Q = sample_branch(self.robot, n_samples, seed)
        X = pose_features(self.robot.fk_batch(Q))
        Y = encode_joints(Q)
        self.scaler = StandardScaler().fit(X)
        self.model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                                  learning_rate_init=1e-3, batch_size=512, max_iter=max_iter,
                                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                                  random_state=seed, verbose=verbose)
        self.model.fit(self.scaler.transform(X), Y)
        return self

    def predict(self, T) -> np.ndarray:
        """Predict joint angles for one (4,4) pose or a batch (N,4,4)."""
        single = np.asarray(T).ndim == 2
        X = pose_features(np.asarray(T)[None] if single else np.asarray(T))
        Q = decode_joints(self.model.predict(self.scaler.transform(X)))
        return Q[0] if single else Q

    def solve(self, T_target, **ik_kwargs):
        """Hybrid solve: network guess -> damped-least-squares refinement."""
        q0 = self.predict(T_target)
        return self.robot.ik_numeric(T_target, q0=q0, **ik_kwargs)

    def save(self, path):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path, robot: DHRobot | None = None):
        d = joblib.load(path)
        return cls(robot or DHRobot(), d["model"], d["scaler"])


def benchmark(nik: NeuralIK, n_targets=500, seed=123, max_iter=100):
    """Compare IK strategies on random reachable targets from the training branch."""
    robot = nik.robot
    rng = np.random.default_rng(seed)
    Q_true = sample_branch(robot, n_targets, seed)
    targets = robot.fk_batch(Q_true)
    home = np.array([0, 0, 0, 0, 30, 0]) * DEG

    strategies = {
        "DLS from home pose": lambda T: robot.ik_numeric(T, q0=home, max_iter=max_iter),
        "DLS from random seed": lambda T: robot.ik_numeric(T, q0=rng.uniform(BRANCH_MIN, BRANCH_MAX), max_iter=max_iter),
        "Neural net + DLS (hybrid)": lambda T: nik.solve(T, max_iter=max_iter),
    }
    results = {}

    # Network on its own
    t0 = time.perf_counter()
    Q_nn = nik.predict(targets)
    dt = (time.perf_counter() - t0) / n_targets
    P_nn = robot.fk_batch(Q_nn)[:, :3, 3]
    pos_err = np.linalg.norm(P_nn - targets[:, :3, 3], axis=1)
    results["Neural net only"] = {"success_rate": float(np.mean(pos_err < 1.0)),
                                  "median_pos_err_mm": float(np.median(pos_err)),
                                  "mean_iters": 0.0, "ms_per_solve": 1000 * dt}

    for name, fn in strategies.items():
        ok, iters, errs = [], [], []
        t0 = time.perf_counter()
        for T in targets:
            q, success, it = fn(T)
            ok.append(success)
            iters.append(it)
            errs.append(np.linalg.norm(robot.fk(q)[:3, 3] - T[:3, 3]))
        dt = (time.perf_counter() - t0) / n_targets
        results[name] = {"success_rate": float(np.mean(ok)), "median_pos_err_mm": float(np.median(errs)),
                         "mean_iters": float(np.mean(iters)), "ms_per_solve": 1000 * dt}
    return results
