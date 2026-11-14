import numpy as np
import pytest

from ml.fault_detection import (DT, K_OBS, debounce, detect, flange_point_jacobian, momentum_observer,
                                naive_residual, observer_filter, random_moves, simulate_fleet)
from robotics.dynamics import ArmDynamics

dyn = ArmDynamics()


@pytest.fixture(scope="module")
def fleet():
    return simulate_fleet(4, ["none", "collision", "payload", "actuator"], seed=7)


def test_flange_jacobian_matches_kinematics():
    q = dyn.robot.random_q(4, rng=0)
    J, p = flange_point_jacobian(dyn, q, (0, 0, 0))
    for i in range(4):
        assert np.allclose(J[i], dyn.robot.jacobian(q[i])[:3] / 1000, atol=1e-12)
        assert np.allclose(p[i] * 1000, dyn.robot.fk(q[i])[:3, 3])


def test_quintic_moves_are_rest_to_rest_and_consistent():
    q, qd, qdd = random_moves(5, np.random.default_rng(0), dyn.robot)
    assert np.allclose(qd[0], 0) and np.allclose(qd[-1], 0, atol=1e-3)
    assert np.allclose(np.diff(q, axis=0) / DT, qd[:-1], atol=2e-3)


def test_observer_filter_is_first_order():
    y = observer_filter(np.ones((int(1.0 / DT), 1)))
    i_tau = int(1 / K_OBS / DT)
    assert y[i_tau, 0] == pytest.approx(1 - np.exp(-1), abs=0.02)


def test_debounce_requires_consecutive_samples():
    s = np.zeros((50, 2))
    s[10:14, 0] = 5          # 4-sample blip
    s[20:40, 1] = 5          # sustained
    d = debounce(s, k=10)
    hit, first = detect(d, 1.0)
    assert list(hit) == [False, True] and first[1] == 29


def test_fleet_tracks_and_faults_show_in_observer(fleet):
    """Each faulty arm vs its fault-free twin (same unit, trajectory and noise): the observer
    residuals agree until the fault starts, then separate."""
    assert np.abs(fleet.q - fleet.q_des).max() < 0.06            # rad, without friction compensation
    twin = simulate_fleet(4, ["none"] * 4, seed=7, units=fleet.units)
    assert np.allclose(twin.q_des, fleet.q_des)
    R, Rt = momentum_observer(fleet, dyn), momentum_observer(twin, dyn)
    for n in (1, 2, 3):
        i = int(fleet.onset[n] / DT)
        assert np.abs(R[:i - 1, n] - Rt[:i - 1, n]).max() < 1e-6
        # collisions and payload change load the big joints; a weakened motor on a lightly loaded
        # joint changes the torque balance only a little (why actuator faults are hardest)
        assert np.abs(R[i:, n] - Rt[i:, n]).max() > (2.0 if fleet.fault[n] != "actuator" else 0.05)


def test_observer_and_naive_see_the_same_friction_with_opposite_sign(fleet):
    """On a normal run both residuals are dominated by unmodelled friction. The observer estimates
    torque acting *on* the arm (friction opposes motion: -f), the naive residual is the *extra*
    torque the motor supplied (+f), so they are strongly anti-correlated on the big joints."""
    R, Nv = momentum_observer(fleet, dyn)[200:, 0], naive_residual(fleet, dyn)[200:, 0]
    for j in range(3):
        assert np.corrcoef(R[:, j], Nv[:, j])[0, 1] < -0.8


def test_unit_calibration_recovers_known_coefficients(fleet):
    from ml.fault_detection import UnitCalibration
    cal = UnitCalibration(dyn, filtered=True, ridge=1e-9)
    B = cal.basis(fleet)                                     # (steps, N, 6, 4), already filtered
    w_true = np.random.default_rng(0).normal(size=(fleet.q.shape[1], 6, 4))
    resid = np.einsum("tnjk,njk->tnj", B, w_true)
    cal.fit(fleet, resid)
    # some weights are unidentifiable (joint 1 rotates about vertical, so payload gravity is
    # identically zero there); what matters is that the fitted model reproduces the residual
    assert np.abs(cal.apply(fleet, resid)).max() < 1e-3 * np.abs(resid).max()


def test_units_rerun_same_physical_arm():
    from ml.fault_detection import Units
    a = simulate_fleet(2, seed=3)
    b = simulate_fleet(2, seed=4, units=a.units)
    assert np.allclose(a.units.visc, b.units.visc) and np.allclose(a.units.strib, b.units.strib)
    assert not np.allclose(a.q_des, b.q_des)                 # new trajectory
    both = Units.concat([a.units, b.units[0:1]])
    assert both.visc.shape == (3, 6)
