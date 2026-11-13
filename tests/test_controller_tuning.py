import numpy as np

from ml.bayes_opt import BayesianOptimizer, expected_improvement, random_search
from ml.controller_tuning import (BOUNDS, K, M, closed_loop_poly, linear_stable, make_scenarios,
                                  nominal_scenario, classical_baseline_gains, robust_cost, simulate)
from robotics.control import routh_stable


def test_expected_improvement_prefers_low_mean_and_high_uncertainty():
    assert expected_improvement(np.array([0.0]), np.array([0.1]), 1.0) > \
        expected_improvement(np.array([0.9]), np.array([0.1]), 1.0)
    assert expected_improvement(np.array([1.0]), np.array([1.0]), 1.0) > \
        expected_improvement(np.array([1.0]), np.array([0.01]), 1.0)


def test_bo_finds_minimum_of_log_quadratic_and_beats_random():
    target = np.array([3.0, 0.2])
    f = lambda x: 1e-3 + np.sum(np.log10(x / target) ** 2)  # noqa: E731
    bounds = [(0.01, 100.0), (0.01, 100.0)]
    bo = BayesianOptimizer(bounds, seed=0).minimize(f, n_init=6, n_iter=20)
    rs = random_search(f, bounds, 26, seed=0)
    assert np.allclose(np.log10(bo.x_best), np.log10(target), atol=0.15)
    assert bo.y_best < rs.y_best
    assert np.all(np.diff(bo.best_so_far) <= 0)


def test_unit_cube_roundtrip():
    bo = BayesianOptimizer(BOUNDS["pid"])
    x = np.array([10.0, 5.0, 1.0])
    assert np.allclose(bo.to_param(bo.to_unit(x)), x)


def test_partitioned_baseline_exact_on_nominal_plant():
    """With a perfect model the partitioned loop is the designed critically damped 2nd-order system."""
    t, X, _ = simulate("partitioned", classical_baseline_gains(), nominal_scenario())
    g = classical_baseline_gains()
    wn = np.sqrt(g[0])
    analytic = 1 - (1 + wn * t) * np.exp(-wn * t)
    assert np.max(np.abs(X[:, 0] - analytic)) < 5e-3
    assert X[:, 0].max() <= 1.0 + 1e-6     # no overshoot (zeta = 1)


def test_baseline_unstable_when_spring_overestimated():
    """Net stiffness M kp + k_true - K < 0 once k_true < K - M kp (k overestimated by > 25%)."""
    g = classical_baseline_gains()
    k_crit = K - M * g[0]
    assert not routh_stable(closed_loop_poly("partitioned", g, M, 1.0, 0.95 * k_crit))
    assert routh_stable(closed_loop_poly("partitioned", g, M, 1.0, 1.05 * k_crit))


def test_linear_stability_matches_simulation():
    sc = make_scenarios(40, seed=3)
    stable = linear_stable("partitioned", classical_baseline_gains(), sc)
    _, X, _ = simulate("partitioned", classical_baseline_gains(), sc)
    assert (~stable).any() and stable.any()
    # unstable loops run away (until saturation), stable ones stay bounded near the target
    assert np.all(np.abs(X[-1, ~stable]) > np.abs(X[-1, stable]).max())


def test_tuned_gains_beat_classical_baseline_on_robust_cost():
    sc = make_scenarios(16, seed=1)
    tuned = robust_cost("pid", [12.0, 8.7, 5.6], sc)
    assert tuned < 0.5 * robust_cost("partitioned", classical_baseline_gains(), sc)


def test_saturation_respected():
    _, _, U = simulate("pid", [300.0, 200.0, 200.0], make_scenarios(4, seed=2))
    assert np.abs(U).max() <= 30.0 + 1e-9
