"""Phase 3: learn unmodelled friction / cogging / parameter error with a GP, episode by episode.

    python scripts/learn_dynamics.py      # ~2 min on CPU

Writes results/gp_dynamics.json and results/gp_dynamics.png.
"""
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

from ml.gp_dynamics import ResidualGP, TruePlant, reference, rmse, run  # noqa: E402

EPISODES = 5
plant = TruePlant()
train_ref, test_ref = reference("train"), reference("test")

base_train, base_test = run(plant, train_ref), run(plant, test_ref, seed=100)
oracle_train = run(plant, train_ref, plant_aware=True)
oracle_test = run(plant, test_ref, plant_aware=True, seed=100)

results = {"nominal_model": {"train_rmse_mm": 1e3 * rmse(base_train), "unseen_rmse_mm": 1e3 * rmse(base_test)},
           "oracle_true_model": {"train_rmse_mm": 1e3 * rmse(oracle_train), "unseen_rmse_mm": 1e3 * rmse(oracle_test)}}
curves, gps, logs_test = {}, {}, {}
for kern in ["structured", "rbf"]:
    gp, log = ResidualGP(kern), base_train
    tr, te = [1e3 * rmse(base_train)], [1e3 * rmse(base_test)]
    for ep in range(1, EPISODES + 1):
        gp.add(log).fit()                    # learn from everything seen so far
        log = run(plant, train_ref, ff=gp.feedforward(train_ref), seed=ep)
        lt = run(plant, test_ref, ff=gp.feedforward(test_ref), seed=100 + ep)
        tr.append(1e3 * rmse(log))
        te.append(1e3 * rmse(lt))
        if ep == 1:
            logs_test[kern] = lt
        print(f"[{kern:10s}] episode {ep}: train {tr[-1]:7.3f} mm   unseen {te[-1]:7.3f} mm")
    curves[kern], gps[kern] = (tr, te), gp
    results[f"gp_{kern}"] = {"train_rmse_mm_by_episode": tr, "unseen_rmse_mm_by_episode": te,
                             "hyperparameters": gp.hyperparameters()}

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "gp_dynamics.json").write_text(json.dumps(results, indent=2))

# ---------------------------------------------------------------- figure
BLUE, ORANGE, GRAY, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.titleweight": "bold", "axes.titlecolor": INK})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=BG)
for a in ax:
    a.set_facecolor(BG)

t = test_ref[0]
ax[0].plot(t, 1e3 * base_test["e"], color=GRAY, lw=1.2, label="nominal model only")
ax[0].plot(t, 1e3 * logs_test["rbf"]["e"], color=ORANGE, lw=1.2, label="+ black-box GP")
ax[0].plot(t, 1e3 * logs_test["structured"]["e"], color=BLUE, lw=1.6, label="+ physics-structured GP")
ax[0].set(title="Tracking error on an unseen trajectory", xlabel="time (s)", ylabel="error (mm)", ylim=(-38, 58))
ax[0].legend(frameon=False, fontsize=8, loc="upper center", ncol=3)

eps = np.arange(EPISODES + 1)
for kern, col, lab in [("structured", BLUE, "physics-structured GP"), ("rbf", ORANGE, "black-box GP")]:
    ax[1].plot(eps, curves[kern][1], color=col, lw=2, marker="o", ms=5, label=f"{lab}: unseen")
    ax[1].plot(eps, curves[kern][0], color=col, lw=2, ls="--", marker="o", ms=5, mfc=BG, label=f"{lab}: training")
ax[1].axhline(results["oracle_true_model"]["unseen_rmse_mm"], color=INK2, lw=1, ls=":")
ax[1].text(EPISODES, results["oracle_true_model"]["unseen_rmse_mm"] * 1.25, "perfect-model limit",
           color=INK2, fontsize=8, ha="right")
ax[1].set(title="Learning curve", xlabel="learning episodes (20 s runs)", ylabel="RMS error (mm)", yscale="log", ylim=(0.15, 300))
ax[1].legend(frameon=False, fontsize=8, loc="upper right", ncol=2)

v = np.linspace(-1.6, 1.6, 400)
true_r = plant.residual(0 * v, v, 0 * v)
ax[2].plot(v, true_r, color=INK, lw=2, label="true unmodelled force")
for kern, col, lab in [("rbf", ORANGE, "black-box GP"), ("structured", BLUE, "physics-structured GP")]:
    m, s = gps[kern].predict(0 * v, v, 0 * v, return_std=True)
    ax[2].plot(v, m, color=col, lw=1.6, label=f"{lab} (±2σ)")
    ax[2].fill_between(v, m - 2 * s, m + 2 * s, color=col, alpha=0.15, lw=0)
ax[2].set(title="What the GP learned: friction vs velocity", xlabel="velocity (at x = 0, a = 0)",
          ylabel="residual force (N)", ylim=(-8, 8))
ax[2].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "results" / "gp_dynamics.png", dpi=140)
print(json.dumps({k: v for k, v in results.items() if "gp_" not in k}, indent=1))
