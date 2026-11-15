import numpy as np
import pytest

from ml.arm_learning import ArmFeedForward, default_log_gains, simulate_gains, tracking_rms, true_unmodelled_torque
from ml.fault_detection import DT, KD, KP, Units, excitation_moves, random_moves, simulate_fleet
from robotics.dynamics import ArmDynamics

dyn = ArmDynamics()


def test_excitation_trajectory_starts_and_ends_at_rest_and_sweeps_both_ways():
    q, qd, qdd = excitation_moves(3, np.random.default_rng(0), t_run=10.0)
    assert np.abs(qd[0]).max() < 1e-2 and np.abs(qd[-1]).max() < 1e-2
    assert np.all(qd.max(0) > 0.2) and np.all(qd.min(0) < -0.2)        # every joint, both directions
    assert np.allclose(np.gradient(q, DT, axis=0), qd)


@pytest.fixture(scope="module")
def arms():
    units = Units.sample(2, np.random.default_rng(3), dyn)
    ref = random_moves(2, np.random.default_rng(4), dyn.robot, t_run=3.0)
    return units, ref


def test_oracle_feedforward_tracks_almost_perfectly(arms):
    units, ref = arms
    base = simulate_fleet(2, seed=5, units=units, ref=ref, t_run=3.0)
    orc = simulate_fleet(2, seed=5, units=units, ref=ref, t_run=3.0,
                         ff=true_unmodelled_torque(units, dyn, ref[0], ref[1]))
    assert tracking_rms(orc).max() < 2e-4                                   # rad
    assert tracking_rms(base).max() > 10 * tracking_rms(orc).max()


def test_learned_feedforward_from_excitation_generalises(arms):
    units, ref = arms
    learn = simulate_fleet(2, seed=6, units=units, trajectory="excitation", t_run=10.0)
    model = ArmFeedForward(dyn, use_gp=False).fit(learn)
    base = simulate_fleet(2, seed=5, units=units, ref=ref, t_run=3.0)
    lrn = simulate_fleet(2, seed=5, units=units, ref=ref, t_run=3.0, ff=model.predict(ref[0], ref[1]))
    assert np.all(tracking_rms(lrn).mean(0)[3:] < tracking_rms(base).mean(0)[3:] / 5)   # wrist joints


def test_per_joint_gains_match_scalar_defaults(arms):
    units, ref = arms
    a = simulate_fleet(2, seed=5, units=units, ref=ref, t_run=3.0)
    b = simulate_fleet(2, seed=5, units=units, ref=ref, t_run=3.0, kp=np.full(6, KP), kd=np.full(6, KD))
    assert np.array_equal(a.q, b.q)


def test_simulate_gains_batches_candidates_independently(arms):
    units, _ = arms
    g = default_log_gains()
    both = simulate_gains(np.vstack([g, g + 0.3]), units, seed=7)
    one = simulate_gains(g[None], units, seed=7)
    assert np.isclose(both[0][0], one[0][0]) and np.isclose(both[1][0], one[1][0])
    assert both[1][1] > both[1][0]                                           # higher gains, more chatter
