import numpy as np

from ml.controller_tuning import make_scenarios, robust_cost, simulate
from ml.rl_policy import LIN_SCALE, PolicyShape, policy_costs, simulate_policies, train_es


def pid_only_theta(kp, kd, ki, shape=PolicyShape()):
    th = np.zeros(shape.dim)
    th[:3] = np.array([kp, kd, ki]) / LIN_SCALE
    return th


def test_policy_with_zero_network_is_exactly_pid():
    sc = make_scenarios(6, seed=4)
    g = (12.0, 8.0, 5.0)
    X, U = simulate_policies(pid_only_theta(*g)[None], sc)
    _, Xp, Up = simulate("pid", g, sc)
    np.testing.assert_allclose(X[:, 0], Xp, atol=1e-10)
    np.testing.assert_allclose(policy_costs(pid_only_theta(*g)[None], sc)[0], robust_cost("pid", g, sc), rtol=1e-9)


def test_population_is_simulated_independently():
    sc = make_scenarios(4, seed=5)
    a, b = pid_only_theta(10, 5, 2), pid_only_theta(20, 9, 4)
    both = policy_costs(np.vstack([a, b]), sc)
    assert np.isclose(both[0], policy_costs(a[None], sc)[0]) and np.isclose(both[1], policy_costs(b[None], sc)[0])


def test_es_improves_cost():
    sc = make_scenarios(4, seed=6)
    _, best, hist = train_es(sc, generations=8, pop_half=6, verbose=False)
    assert best < hist[0]
