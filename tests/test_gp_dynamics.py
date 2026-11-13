import numpy as np
import pytest

from ml.gp_dynamics import ResidualGP, TruePlant, reference, rmse, run

plant = TruePlant()


@pytest.fixture(scope="module")
def learned():
    log = run(plant, reference("train"))
    return log, ResidualGP("structured", max_points=300).add(log).fit()


def test_oracle_controller_tracks_almost_perfectly():
    assert rmse(run(plant, reference("test"), plant_aware=True)) < 1e-3          # < 1 mm
    assert rmse(run(plant, reference("test"))) > 10 * rmse(run(plant, reference("test"), plant_aware=True))


def test_residual_target_matches_true_unmodelled_force(learned):
    log, _ = learned
    from ml.gp_dynamics import B, K, M
    r_logged = log["f"] - (M * log["a"] + B * log["v"] + K * log["x"])
    r_true = plant.residual(log["x"], log["v"], log["a"])
    # differs only by measurement noise on a (scaled by the true mass)
    assert np.std(r_logged - r_true) < 1.0


def test_structured_gp_recovers_friction_curve(learned):
    _, gp = learned
    v = np.linspace(-0.8, 0.8, 50)
    v = v[np.abs(v) > 0.1]                         # away from the Coulomb step
    err = gp.predict(0 * v, v, 0 * v) - plant.residual(0 * v, v, 0 * v)
    assert np.sqrt(np.mean(err ** 2)) < 0.5        # N, vs ~2-4 N of friction


def test_gp_feedforward_generalises_to_unseen_trajectory(learned):
    _, gp = learned
    ref = reference("test")
    base, ff = rmse(run(plant, ref, seed=5)), rmse(run(plant, ref, ff=gp.feedforward(ref), seed=5))
    assert ff < base / 5


def test_uncertainty_grows_away_from_data(learned):
    _, gp = learned
    _, s_in = gp.predict(np.array([0.0]), np.array([0.3]), np.array([0.0]), return_std=True)
    _, s_out = gp.predict(np.array([0.0]), np.array([5.0]), np.array([0.0]), return_std=True)
    assert s_out[0] > 3 * s_in[0]
