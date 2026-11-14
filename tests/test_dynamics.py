import numpy as np
import pytest

from robotics.dynamics import ArmDynamics

dyn = ArmDynamics()
Q = dyn.robot.random_q(8, rng=0)


def test_mass_matrix_symmetric_positive_definite():
    M = dyn.mass_matrix(Q)
    assert np.allclose(M, M.swapaxes(1, 2), atol=1e-12)
    assert np.linalg.eigvalsh(M).min() > 0


def test_gravity_is_gradient_of_potential_energy():
    for q in Q[:3]:
        fd = np.array([(dyn.potential_energy(q + 1e-6 * e) - dyn.potential_energy(q - 1e-6 * e)) / 2e-6
                       for e in np.eye(6)])
        assert np.allclose(dyn.gravity(q)[0], fd, atol=1e-5)


def test_rnea_is_linear_in_acceleration():
    rng = np.random.default_rng(1)
    qd, qdd = rng.normal(size=(8, 6)), rng.normal(size=(8, 6))
    tau = dyn.rnea(Q, qd, qdd)
    M = dyn.mass_matrix(Q)
    assert np.allclose(tau, np.einsum("nij,nj->ni", M, qdd) + dyn.bias(Q, qd, friction=False), atol=1e-9)


def test_coriolis_does_no_work():
    """qd^T (C qd) = 1/2 qd^T Mdot qd (skew-symmetry of Mdot - 2C)."""
    rng = np.random.default_rng(2)
    q, qd = Q[0], rng.normal(size=6)
    c = dyn.rnea(q, qd, np.zeros(6), gravity=False)[0]
    h = 1e-6
    Mdot = (dyn.mass_matrix(q + h * qd)[0] - dyn.mass_matrix(q - h * qd)[0]) / (2 * h)
    assert qd @ c == pytest.approx(0.5 * qd @ Mdot @ qd, rel=1e-5)


def test_energy_conserved_in_free_fall():
    def f(s):
        return np.r_[s[6:], dyn.forward(s[:6], s[6:], np.zeros(6), friction=False)[0]]
    s = np.r_[Q[1], np.zeros(6)]
    E0 = dyn.potential_energy(s[:6])
    dt = 1e-3
    for _ in range(300):
        k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)  # noqa: E702
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    E1 = dyn.potential_energy(s[:6]) + dyn.kinetic_energy(s[:6], s[6:])
    assert dyn.kinetic_energy(s[:6], s[6:]) > 1.0          # it really moved
    assert abs(E1 - E0) < 1e-5


def test_gravity_free_wrist_roll_and_base_yaw():
    """Joint 1 rotates about vertical, so gravity never loads it; the stretched arm loads joint 2."""
    g = dyn.gravity(Q)
    assert np.allclose(g[:, 0], 0, atol=1e-9)
    assert np.abs(g[:, 1]).max() > 50


def test_payload_increases_gravity_load():
    heavy = ArmDynamics(payload_kg=10.0)
    q = np.array([0, 0.3, 0.2, 0, 0.5, 0])
    assert abs(heavy.gravity(q)[0, 1]) > abs(dyn.gravity(q)[0, 1])


def test_batched_equals_single():
    rng = np.random.default_rng(3)
    qd, qdd = rng.normal(size=(8, 6)), rng.normal(size=(8, 6))
    batch = dyn.rnea(Q, qd, qdd, friction=True)
    single = np.vstack([dyn.rnea(Q[i], qd[i], qdd[i], friction=True) for i in range(8)])
    assert np.allclose(batch, single)
