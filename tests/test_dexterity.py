import numpy as np
import pytest

from ml.dexterity import DexterityMap, drawing_probe_points, place_points, soft_min
from pathgen import sources
from pathgen.rapid import to_rapid
from pathgen.reach import Workcell, check_reach
from robotics.kinematics import DHRobot

robot = DHRobot()


def test_dexterity_index_bounds_and_singularity():
    Q = robot.random_q(200, rng=1)
    d = np.array([robot.dexterity(q) for q in Q])
    assert np.all((d >= 0) & (d <= 1))
    q = Q[0].copy()
    q[4] = 0.0                          # wrist singularity: axes 4 and 6 align
    assert robot.dexterity(q) < 1e-6


def test_dexterity_is_independent_of_joint6():
    q = robot.random_q(1, rng=2)[0]
    q2 = q.copy()
    q2[5] += 1.3
    assert np.isclose(robot.dexterity(q), robot.dexterity(q2))


def test_yawed_workcell_keeps_pen_down_and_rotates_paper():
    cell = Workcell(paper_origin_mm=(800.0, 0.0, 300.0), yaw_deg=90.0)
    T = cell.flange_pose((100.0, 0.0))
    assert np.allclose(T[:3, 2], [0, 0, -1])                    # tool z still points down
    tip = T[:3, 3] + T[:3, :3] @ np.array([0, 0, cell.pen_length_mm])
    assert np.allclose(tip, [800.0, 100.0, 300.0])              # paper x-axis now along base y


def test_rapid_wobj_quaternion_matches_yaw():
    code = to_rapid(sources.shape_strokes("square", 20), yaw_deg=90.0)
    assert "[0.707107,0,0,0.707107]" in code


def test_yawed_placement_still_reachable():
    stroke = [np.array([[0.0, 0.0], [60.0, 40.0]])]
    assert check_reach(stroke, Workcell(paper_origin_mm=(900.0, 300.0, 300.0), yaw_deg=45.0)).ok


def test_probe_points_and_placement_geometry():
    strokes = sources.shape_strokes("circle", 100)
    P = drawing_probe_points(strokes)
    assert P[:, 0].min() >= -1e-9 and P[:, 0].max() <= 100 + 1e-9
    W = place_points(np.array([[10.0, 0.0]]), np.array([500.0, 0.0, 200.0]), np.pi / 2)
    assert np.allclose(W, [[500.0, 10.0, 200.0]])


def test_soft_min_approaches_min():
    x = np.array([[0.2, 0.25, 0.3]])
    assert soft_min(x, tau=1e-3)[0] == pytest.approx(0.2, abs=2e-3)
    assert soft_min(x)[0] <= x.mean()


def test_saved_map_agrees_with_real_ik():
    dmap = DexterityMap.load()
    from ml.dexterity import label_points
    rng = np.random.default_rng(11)
    P = np.c_[rng.uniform(300, 1400, 40), rng.uniform(-800, 800, 40), rng.uniform(0, 600, 40)]
    reach, dex = label_points(P)
    pr, pd = dmap.predict(P)
    assert np.mean((pr > 0.5) == reach) >= 0.95
    assert np.mean(np.abs(pd[reach] - dex[reach])) < 0.02
