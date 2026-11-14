"""Learned dexterity map + automatic workpiece placement.

Where should the paper (or any workpiece) go? Checking one candidate placement properly means
solving IK for every target on it, about 1 ms per point. Searching thousands of placements that
way is minutes of compute. Instead:

1. Label ~25k random tool-down points in the cell with the real hybrid IK: reachable or not, and
   the dexterity of the solution (inverse condition number of the length-scaled Jacobian, 0..1).
2. Fit gradient-boosted trees: a classifier for reachability and a regressor for dexterity.
3. Score thousands of candidate placements (x, y, z, yaw) on a grid of points covering the
   drawing in one batched prediction. Maximise a soft-min of predicted dexterity with every
   point confidently reachable.
4. Verify the top few with real IK (pathgen.reach) and keep the best verified one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from robotics.kinematics import DHRobot

R_DOWN = np.diag([1.0, -1.0, -1.0])
ROOT = Path(__file__).resolve().parents[1]
BOX = np.array([[-200.0, 1800.0], [-1600.0, 1600.0], [-400.0, 1400.0]])  # sampling box (mm)


def features(P):
    """(N, 3) tool-tip positions -> features. Radius and height carry most of the structure."""
    P = np.atleast_2d(P)
    r = np.hypot(P[:, 0], P[:, 1])
    return np.c_[P, r, np.arctan2(P[:, 1], P[:, 0])]


def label_points(P, pen_length_mm=120.0, robot=None, nik=None, max_iter=100):
    """Real IK labels: reachable (bool) and dexterity in [0, 1]."""
    from pathgen.reach import _load_nik
    robot = robot or DHRobot()
    nik = nik or _load_nik(robot)
    reach, dex = np.zeros(len(P), bool), np.zeros(len(P))
    for i, p in enumerate(P):
        T = np.eye(4)
        T[:3, :3] = R_DOWN
        T[:3, 3] = p - R_DOWN @ np.array([0, 0, pen_length_mm])
        q, ok, _ = nik.solve(T, max_iter=max_iter)
        if ok:
            reach[i] = True
            dex[i] = robot.dexterity(q)
    return reach, dex


@dataclass
class DexterityMap:
    clf: HistGradientBoostingClassifier
    reg: HistGradientBoostingRegressor
    pen_length_mm: float = 120.0

    @classmethod
    def fit(cls, n=25_000, seed=0, pen_length_mm=120.0, verbose=True):
        rng = np.random.default_rng(seed)
        P = BOX[:, 0] + rng.random((n, 3)) * (BOX[:, 1] - BOX[:, 0])
        t0 = time.time()
        reach, dex = label_points(P, pen_length_mm)
        if verbose:
            print(f"  labelled {n} points with real IK in {time.time() - t0:.0f} s "
                  f"({reach.mean():.0%} reachable)")
        X = features(P)
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, random_state=seed).fit(X, reach)
        reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, random_state=seed).fit(X[reach], dex[reach])
        return cls(clf, reg, pen_length_mm)

    def predict(self, P):
        X = features(P)
        return self.clf.predict_proba(X)[:, 1], np.clip(self.reg.predict(X), 0.0, 1.0)

    def save(self, path):
        joblib.dump({"clf": self.clf, "reg": self.reg, "pen": self.pen_length_mm}, path, compress=3)

    @classmethod
    def load(cls, path=ROOT / "models" / "dexterity_map.joblib"):
        d = joblib.load(path)
        return cls(d["clf"], d["reg"], d["pen"])


def drawing_probe_points(strokes, n_grid=6):
    """Points that represent the drawing: bounding-box grid plus samples along the strokes."""
    pts = np.vstack(strokes)
    lo, hi = pts.min(0), pts.max(0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n_grid), np.linspace(lo[1], hi[1], n_grid))
    along = pts[np.linspace(0, len(pts) - 1, min(len(pts), 40)).astype(int)]
    return np.vstack([np.c_[gx.ravel(), gy.ravel()], along])


def place_points(xy, origin, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return np.c_[xy @ R.T + origin[:2], np.full(len(xy), origin[2])]


def soft_min(x, axis=-1, tau=0.01):
    return -tau * np.log(np.mean(np.exp(-x / tau), axis=axis))


def optimise_placement(strokes, dmap: DexterityMap, n_candidates=6000, z_range=(0.0, 600.0),
                       verify_top=5, seed=0, p_min=0.98, verbose=True):
    """Search (x, y, z, yaw) for the paper origin; returns best IK-verified Workcell + diagnostics."""
    from pathgen.reach import Workcell, check_reach
    rng = np.random.default_rng(seed)
    probe = drawing_probe_points(strokes)
    C = np.c_[rng.uniform(200, 1500, n_candidates), rng.uniform(-900, 900, n_candidates),
              rng.uniform(*z_range, n_candidates), rng.uniform(-np.pi, np.pi, n_candidates)]
    t0 = time.time()
    P = np.vstack([place_points(probe, c[:3], c[3]) for c in C])
    p_reach, dex = dmap.predict(P)
    p_reach, dex = p_reach.reshape(n_candidates, -1), dex.reshape(n_candidates, -1)
    score = np.where(p_reach.min(1) >= p_min, soft_min(dex), -np.inf)
    t_surrogate = time.time() - t0
    order = np.argsort(-score)[:verify_top]
    verified = []
    t1 = time.time()
    for i in order:
        if not np.isfinite(score[i]):
            continue
        cell = Workcell(paper_origin_mm=tuple(float(v) for v in C[i, :3]), yaw_deg=float(np.degrees(C[i, 3])),
                        pen_length_mm=dmap.pen_length_mm)
        rep = check_reach(strokes, cell)
        verified.append((rep.ok, rep.min_dexterity, cell, rep, float(score[i])))
        if verbose:
            print(f"  candidate {tuple(np.round(C[i, :3]))}, yaw {np.degrees(C[i, 3]):6.1f} deg: "
                  f"predicted {score[i]:.3f}, verified {'OK' if rep.ok else 'FAIL'} "
                  f"min dexterity {rep.min_dexterity:.3f}")
    t_verify = time.time() - t1
    ok = [v for v in verified if v[0]]
    best = max(ok, key=lambda v: v[1]) if ok else None
    return {"cell": best[2] if best else None, "report": best[3] if best else None,
            "predicted_score": best[4] if best else None, "n_candidates": n_candidates,
            "n_probe_points": len(probe), "surrogate_seconds": t_surrogate, "verify_seconds": t_verify,
            "n_verified_ok": len(ok), "n_verified": len(verified)}
