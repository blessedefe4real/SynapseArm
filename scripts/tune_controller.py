"""Phase 2: Bayesian-optimised robust controller tuning vs a textbook computed-torque design.

    python scripts/tune_controller.py              # ~5 min on CPU
    python scripts/tune_controller.py --seeds 1    # quick run

Tunes on 24 training scenarios, then scores every design on 200 *held-out* scenarios.
Writes results/controller_tuning.json and results/controller_tuning.png.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.bayes_opt import BayesianOptimizer, random_search  # noqa: E402
from ml.controller_tuning import (BOUNDS, T_DIST, X_REF, make_scenarios, nominal_scenario,  # noqa: E402
                                  classical_baseline_gains, robust_cost, simulate, summarise)

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--n-init", type=int, default=8)
p.add_argument("--n-iter", type=int, default=32)
args = p.parse_args()
budget = args.n_init + args.n_iter

train = make_scenarios(24, seed=0)
test = make_scenarios(200, seed=999)

designs = {"Classical: computed-torque PD": ("partitioned", classical_baseline_gains())}
curves = {}
for kind, label in [("partitioned", "BO: partitioned + integral"), ("pid", "BO: model-free PID")]:
    f = lambda g, kind=kind: robust_cost(kind, g, train)  # noqa: E731
    bo_runs, rs_runs = [], []
    for s in range(args.seeds):
        print(f"[{kind}] seed {s}: Bayesian optimisation ({budget} evals)...")
        bo_runs.append(BayesianOptimizer(BOUNDS[kind], seed=s).minimize(f, args.n_init, args.n_iter))
        print(f"[{kind}] seed {s}: random search ({budget} evals)...")
        rs_runs.append(random_search(f, BOUNDS[kind], budget, seed=s))
        print(f"    BO best {bo_runs[-1].y_best:.3f}   random best {rs_runs[-1].y_best:.3f}")
    best = min(bo_runs, key=lambda r: r.y_best)
    designs[label] = (kind, best.x_best)
    curves[kind] = {"bo": np.array([r.best_so_far for r in bo_runs]),
                    "rs": np.array([r.best_so_far for r in rs_runs])}

results = {"held_out_test": {}, "nominal_plant": {}, "optimiser_comparison": {}}
for name, (kind, g) in designs.items():
    results["held_out_test"][name] = summarise(kind, g, test)
    results["nominal_plant"][name] = summarise(kind, g, nominal_scenario())
for kind, c in curves.items():
    results["optimiser_comparison"][kind] = {
        "budget": budget,
        "bo_final_mean": float(c["bo"][:, -1].mean()), "rs_final_mean": float(c["rs"][:, -1].mean()),
        # evaluations BO needs to beat random search's final result
        "bo_evals_to_match_random": int(np.argmax(c["bo"].mean(0) <= c["rs"][:, -1].mean()) + 1),
    }

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "controller_tuning.json").write_text(json.dumps(results, indent=2))

# ---------------------------------------------------------------- figure
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
colors = dict(zip(designs, [GRAY, BLUE, ORANGE]))
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.titleweight": "bold", "axes.titlecolor": INK})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), facecolor="#fcfcfb")
for a in ax:
    a.set_facecolor("#fcfcfb")

# (a) responses on 30 held-out plants
sub = make_scenarios(30, seed=999)
for name, (kind, g) in designs.items():
    t, X, _ = simulate(kind, g, sub)
    ax[0].plot(t, X, color=colors[name], lw=0.8, alpha=0.35)
    ax[0].plot([], [], color=colors[name], lw=2, label=name)
ax[0].axhline(X_REF, color=INK2, lw=1, ls="--")
ax[0].axvline(T_DIST, color=INK2, lw=0.8, ls=":")
ax[0].text(T_DIST + 0.2, 0.05, "load disturbance", color=INK2, fontsize=8)
ax[0].set(title="Step response on 30 unseen plants (±30% m, b, k)", xlabel="time (s)",
          ylabel="position x", ylim=(0, 2.0))
ax[0].legend(frameon=False, fontsize=8, loc="upper right")

# (b) robust cost on held-out test
names = list(designs)
vals = [results["held_out_test"][n]["mean_iae"] for n in names]
bars = ax[1].bar(range(3), vals, color=[colors[n] for n in names], width=0.6, edgecolor="#fcfcfb", lw=2)
for b_, v in zip(bars, vals):
    ax[1].text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", color=INK)
ax[1].set_xticks(range(3), ["Classical\ncomputed-torque PD", "BO partitioned\n+ integral", "BO model-free\nPID"])
ax[1].set(title="Tracking error on 200 held-out plants", ylabel="mean IAE (lower is better)")
ax[1].grid(axis="x", visible=False)

# (c) sample efficiency
evals = np.arange(1, budget + 1)
for kind, col, lab in [("partitioned", BLUE, "partitioned + integral"), ("pid", ORANGE, "model-free PID")]:
    c = curves[kind]
    ax[2].plot(evals, c["bo"].mean(0), color=col, lw=2, label=f"Bayesian opt: {lab}")
    ax[2].plot(evals, c["rs"].mean(0), color=col, lw=2, ls="--", label=f"random search: {lab}")
ax[2].axvline(args.n_init, color=INK2, lw=0.8, ls=":")
ax[2].text(args.n_init + 0.5, ax[2].get_ylim()[1] * 0.9, "GP takes over", color=INK2, fontsize=8)
ax[2].set(title=f"Optimiser efficiency (mean of {args.seeds} seeds)", xlabel="simulations used",
          ylabel="best robust cost so far", yscale="log")
ax[2].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "results" / "controller_tuning.png", dpi=140)

print(f"\n{'Design':42s} {'mean IAE':>9s} {'worst':>7s} {'overshoot':>10s} {'settle':>7s} {'final err':>10s} {'unstable':>9s}")
for n, r in results["held_out_test"].items():
    print(f"{n:42s} {r['mean_iae']:9.3f} {r['worst_iae']:7.3f} {r['mean_overshoot_pct']:9.1f}% "
          f"{r['median_settle_s']:7.2f} {r['mean_final_err']:10.4f} {r['unstable_frac']:9.1%}")
print(json.dumps(results["optimiser_comparison"], indent=1))
