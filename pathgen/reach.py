"""Check a drawing is executable before it ever reaches the robot.

Every pen point is mapped into the robot base frame (paper work-object + pen tool, pen pointing
down), solved with the neural-IK + DLS hybrid from phase 1, then followed with continuation
(each solve seeded by the previous joint vector) so the joint path stays on one branch.
Reports unreachable points, minimum manipulability (closeness to singularity) and the largest
joint jump between consecutive targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robotics.kinematics import DHRobot

R_DOWN = np.diag([1.0, -1.0, -1.0])        # tool z-axis pointing down (180 deg about x)
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Workcell:
    paper_origin_mm: tuple = (700.0, -150.0, 250.0)   # paper bottom-left in robot base frame
    pen_length_mm: float = 120.0
    robot: DHRobot = field(default_factory=DHRobot)
    yaw_deg: float = 0.0                              # paper rotation about the base z-axis

    @property
    def R_paper(self) -> np.ndarray:
        c, s = np.cos(np.radians(self.yaw_deg)), np.sin(np.radians(self.yaw_deg))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def flange_pose(self, xy, z_up=0.0) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R_paper @ R_DOWN
        tip = np.asarray(self.paper_origin_mm) + self.R_paper @ np.array([xy[0], xy[1], z_up])
        T[:3, 3] = tip - R_DOWN @ np.array([0, 0, self.pen_length_mm])
        return T


@dataclass
class ReachReport:
    n_points: int
    n_unreachable: int
    min_manipulability: float          # relative to the median over the robot's workspace (1.0 = typical)
    min_dexterity: float               # inverse condition number of the scaled Jacobian, 0..1
    max_joint_step_deg: float
    q_path: np.ndarray
    neural_seeded: int

    @property
    def ok(self):
        return self.n_unreachable == 0

    def as_dict(self):
        return {"points": self.n_points, "unreachable": self.n_unreachable,
                "min_manipulability_vs_workspace_median": self.min_manipulability,
                "min_dexterity": self.min_dexterity,
                "max_joint_step_deg": self.max_joint_step_deg, "neural_ik_seeds_used": self.neural_seeded}


def _load_nik(robot):
    from ml.neural_ik import NeuralIK
    path = ROOT / "models" / "neural_ik.joblib"
    return NeuralIK.load(path, robot) if path.exists() else None


_MANIP_CACHE: dict = {}


def _workspace_median_manipulability(robot, n=400):
    key = id(robot)
    if key not in _MANIP_CACHE:
        Q = robot.random_q(n, rng=0)
        _MANIP_CACHE[key] = float(np.median([robot.manipulability(q) for q in Q]))
    return _MANIP_CACHE[key]


def check_reach(strokes, cell: Workcell | None = None, z_up=15.0, nik=None) -> ReachReport:
    cell = cell or Workcell()
    robot = cell.robot
    nik = nik if nik is not None else _load_nik(robot)
    targets = []
    for s in strokes:                       # pen-up approach, the stroke, pen-up retract
        targets.append((s[0], z_up))
        targets += [(p, 0.0) for p in s]
        targets.append((s[-1], z_up))
    q_prev, Q, bad, seeded = None, [], 0, 0
    for xy, z in targets:
        T = cell.flange_pose(xy, z)
        ok = False
        if q_prev is not None:
            q, ok, _ = robot.ik_numeric(T, q0=q_prev, max_iter=60)
        if not ok:
            seeded += 1
            q, ok, _ = nik.solve(T, max_iter=100) if nik is not None else robot.ik_numeric(T, max_iter=200)
        if ok:
            q_prev = q
            Q.append(q)
        else:
            bad += 1
    Q = np.array(Q)
    manip = min(robot.manipulability(q) for q in Q[:: max(len(Q) // 200, 1)]) if len(Q) else 0.0
    manip /= _workspace_median_manipulability(robot)
    dex = min(robot.dexterity(q) for q in Q[:: max(len(Q) // 200, 1)]) if len(Q) else 0.0
    step = float(np.degrees(np.abs(np.diff(Q, axis=0)).max())) if len(Q) > 1 else 0.0
    return ReachReport(len(targets), bad, float(manip), float(dex), step, Q, seeded)
