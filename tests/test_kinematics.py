import numpy as np
import pytest

from robotics.control import type1_plant_char_poly, partitioned_pd_gains, routh_stable
from robotics.kinematics import DHRobot

robot = DHRobot()
rng = np.random.default_rng(0)


def closed_form_position(q):
    """Hand-derived closed-form Px, Py, Pz."""
    c1, s1 = np.cos(q[0]), np.sin(q[0])
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    c23, s23 = np.cos(q[1] + q[2]), np.sin(q[1] + q[2])
    r = 150 + 760 * c2 + 140 * c23 + 795 * s23
    return np.array([c1 * r, s1 * r, 505 + 760 * s2 + 140 * s23 - 795 * c23])


@pytest.mark.parametrize("seed", range(20))
def test_fk_matches_closed_form(seed):
    q = DHRobot().random_q(1, rng=seed)[0]
    np.testing.assert_allclose(robot.fk(q)[:3, 3], closed_form_position(q), atol=1e-9)


def test_fk_batch_matches_single():
    Q = robot.random_q(50, rng=1)
    Tb = robot.fk_batch(Q)
    for q, T in zip(Q, Tb):
        np.testing.assert_allclose(robot.fk(q), T, atol=1e-9)


def test_jacobian_linear_part_matches_finite_difference():
    q = robot.random_q(1, rng=2)[0]
    J = robot.jacobian(q)
    eps = 1e-6
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        dp = (robot.fk(q + dq)[:3, 3] - robot.fk(q - dq)[:3, 3]) / (2 * eps)
        np.testing.assert_allclose(J[:3, i], dp, atol=1e-5)


def test_jacobian_column1_closed_form():
    # Analytic: J_v,1 = [-py, px, 0], J_w,1 = [0, 0, 1]
    q = robot.random_q(1, rng=3)[0]
    p = robot.fk(q)[:3, 3]
    np.testing.assert_allclose(robot.jacobian(q)[:, 0], [-p[1], p[0], 0, 0, 0, 1], atol=1e-9)


@pytest.mark.parametrize("elbow", ["up", "down"])
def test_analytic_position_ik(elbow):
    q = np.array([0.4, 0.3, -0.6 if elbow == "down" else 2.0, 0, 0, 0])
    p = robot.fk(q)[:3, 3]
    q123 = robot.ik_position_analytic(p, elbow=elbow)
    np.testing.assert_allclose(robot.fk(np.r_[q123, 0, 0, 0])[:3, 3], p, atol=1e-6)


def test_numeric_ik_roundtrip():
    q_true = np.deg2rad([20, 30, -20, 10, 40, -30])
    T = robot.fk(q_true)
    q, ok, _ = robot.ik_numeric(T, q0=q_true + 0.2)
    assert ok
    np.testing.assert_allclose(robot.fk(q), T, atol=1e-3)


def test_calc1_stability_range():
    assert routh_stable(type1_plant_char_poly(0.5))
    assert routh_stable(type1_plant_char_poly(100))
    assert not routh_stable(type1_plant_char_poly(-0.5))


def test_calc2_gains():
    g = partitioned_pd_gains(m=7, b=1, k=9)
    assert g["alpha"] == 7
    assert g["w_res"] == pytest.approx(np.sqrt(9 / 7))
    assert g["kp"] == pytest.approx((0.5 * np.sqrt(9 / 7)) ** 2)
    assert g["kd"] == pytest.approx(np.sqrt(9 / 7))
