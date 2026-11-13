"""Phase 2b: train a neural controller with evolution strategies; compare with phase 2 on held-out plants.

    python scripts/train_rl_controller.py                  # ~5 min on CPU
    python scripts/train_rl_controller.py --generations 40 # quick run

Needs results/controller_tuning.json (run scripts/tune_controller.py first).
Writes models/es_policy.npy, results/rl_controller.json, results/rl_controller.png.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from ml.controller_tuning import T_DIST, X_REF, make_scenarios, metrics, simulate, summarise  # noqa: E402
from ml.rl_policy import LIN_SCALE, policy_costs, simulate_policies, train_es  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--generations", type=int, default=150)
p.add_argument("--reuse", action="store_true", help="re-evaluate/plot the saved policy without retraining")
args = p.parse_args()

train, test = make_scenarios(24, seed=0), make_scenarios(200, seed=999)
if args.reuse:
    theta = np.load(ROOT / "models" / "es_policy.npy")
    prev = json.loads((ROOT / "results" / "rl_controller.json").read_text())
    hist, train_cost = np.array(prev["training_cost_curve"]), prev["train_robust_cost"]
else:
    print(f"Training ES policy for {args.generations} generations...")
    theta, train_cost, hist = train_es(train, generations=args.generations)
    (ROOT / "models").mkdir(exist_ok=True)
    np.save(ROOT / "models" / "es_policy.npy", theta)


def summarise_policy(theta, sc):
    X, U = simulate_policies(theta[None], sc)
    t = np.arange(X.shape[0]) * 2e-3
    m = metrics(t, X[:, 0, :], U[:, 0, :])
    return {"robust_cost": float(policy_costs(theta[None], sc)[0]), "mean_iae": float(m["iae"].mean()),
            "worst_iae": float(m["iae"].max()), "mean_overshoot_pct": float(m["overshoot_pct"].mean()),
            "settled_frac": float(np.isfinite(m["settle_s"]).mean()), "mean_final_err": float(m["ss_err"].mean()),
            "mean_control_tv": float(m["tv"].mean()),
            "diverged_frac": float(np.mean(np.abs(X_REF - X[-1, 0]) > 0.5)),
            "pid_part_kp_kd_ki": (LIN_SCALE * theta[:3]).tolist()}


bo = json.loads((ROOT / "results" / "controller_tuning.json").read_text())["held_out_test"]
bo_pid = bo["BO: model-free PID"]["gains_kp_kd_ki"]
res = {"ES neural policy": summarise_policy(theta, test),
       "BO: model-free PID": summarise("pid", bo_pid, test),
       "training_cost_curve": hist.tolist(), "train_robust_cost": train_cost,
       "bo_pid_train_robust_cost": summarise("pid", bo_pid, train)["robust_cost"]}
(ROOT / "results" / "rl_controller.json").write_text(json.dumps(res, indent=2))

# ---------------------------------------------------------------- figure
BLUE, ORANGE, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.titleweight": "bold", "axes.titlecolor": INK})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=BG)
for a in ax:
    a.set_facecolor(BG)
ax[0].plot(np.arange(1, len(hist) + 1), hist, color=BLUE, lw=2, label="ES policy (training plants)")
ax[0].axhline(res["bo_pid_train_robust_cost"], color=ORANGE, lw=1.5, ls="--", label="BO-tuned PID")
ax[0].set(title="Evolution-strategies training", xlabel="generation", ylabel="robust cost", yscale="log")
ax[0].legend(frameon=False, fontsize=8)

sub = make_scenarios(30, seed=999)
t = np.arange(int(20 / 2e-3)) * 2e-3
Xp, _ = simulate_policies(theta[None], sub)
_, Xb, _ = simulate("pid", bo_pid, sub)
ax[1].plot(t, Xb, color=ORANGE, lw=0.8, alpha=0.4)
ax[1].plot(t, Xp[:, 0, :], color=BLUE, lw=0.8, alpha=0.4)
ax[1].plot([], [], color=ORANGE, lw=2, label="BO-tuned PID")
ax[1].plot([], [], color=BLUE, lw=2, label="ES neural policy")
ax[1].axhline(X_REF, color=INK2, lw=1, ls="--")
ax[1].axvline(T_DIST, color=INK2, lw=0.8, ls=":")
ax[1].set(title="30 unseen plants", xlabel="time (s)", ylabel="position x", ylim=(0, 1.5))
ax[1].legend(frameon=False, fontsize=8, loc="lower right")

labels = ["mean IAE", "worst-case IAE"]
es, bp = res["ES neural policy"], res["BO: model-free PID"]
vals_es = [es["mean_iae"], es["worst_iae"]]
vals_bo = [bp["mean_iae"], bp["worst_iae"]]
xx = np.arange(2)
for off, vals, col, lab in [(-0.18, vals_bo, ORANGE, "BO-tuned PID"), (0.18, vals_es, BLUE, "ES neural policy")]:
    bars = ax[2].bar(xx + off, vals, width=0.34, color=col, label=lab, edgecolor=BG, lw=2)
    for b_, v in zip(bars, vals):
        ax[2].text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
ax[2].set_xticks(xx, labels)
ax[2].grid(axis="x", visible=False)
ax[2].set(title="Tracking error, 200 held-out plants", ylabel="IAE (lower is better)", ylim=(0, 3.2))
ax[2].text(0.02, 0.97, f"overshoot: PID {bp['mean_overshoot_pct']:.1f}%  vs  ES {es['mean_overshoot_pct']:.1f}%\n"
           f"settled within 10 s: PID {bp['settled_frac']:.0%}  vs  ES {es['settled_frac']:.0%}",
           transform=ax[2].transAxes, va="top", fontsize=8, color=INK2)
ax[2].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(ROOT / "results" / "rl_controller.png", dpi=140)

print(f"\n{'':22s} {'mean IAE':>9s} {'worst':>7s} {'overshoot':>10s} {'settled':>8s} {'final err':>10s}")
for n in ["BO: model-free PID", "ES neural policy"]:
    r = res[n]
    print(f"{n:22s} {r['mean_iae']:9.3f} {r['worst_iae']:7.3f} {r['mean_overshoot_pct']:9.1f}% "
          f"{r['settled_frac']:8.0%} {r['mean_final_err']:10.4f}")
