"""Learned dexterity map: train, evaluate on held-out IK labels, and demo automatic placement.

    python scripts/train_dexterity_map.py           # ~5 min on CPU (most of it is IK labelling)
    python scripts/train_dexterity_map.py --reuse   # re-evaluate the saved map

Writes models/dexterity_map.joblib, results/dexterity_map.json, results/dexterity_map.png.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from ml.dexterity import BOX, DexterityMap, label_points, optimise_placement  # noqa: E402
from pathgen import optimize, sources  # noqa: E402
from pathgen.reach import Workcell, check_reach  # noqa: E402

if "--reuse" in sys.argv:
    dmap = DexterityMap.load()
else:
    print("Training dexterity map...")
    dmap = DexterityMap.fit(n=25_000)
    dmap.save(ROOT / "models" / "dexterity_map.joblib")

print("Held-out evaluation...")
rng = np.random.default_rng(42)
P = BOX[:, 0] + rng.random((3000, 3)) * (BOX[:, 1] - BOX[:, 0])
reach, dex = label_points(P)
pr, pd = dmap.predict(P)
t0 = time.time()
dmap.predict(np.repeat(P, 100, axis=0))
pred_rate = 300_000 / (time.time() - t0)
ev = {"reach_accuracy": float(np.mean((pr > 0.5) == reach)), "reach_auc": float(roc_auc_score(reach, pr)),
      "dexterity_r2": float(r2_score(dex[reach], pd[reach])),
      "dexterity_mae": float(np.mean(np.abs(dex[reach] - pd[reach]))),
      "surrogate_points_per_s": float(pred_rate)}
print(json.dumps(ev, indent=1))

print("Placement demo on the example sketch...")
strokes = optimize.order_strokes(optimize.simplify(
    sources.normalise(sources.image_strokes(str(ROOT / "examples" / "sketch.png"), width_mm=300), margin_mm=10), 0.1))
default = check_reach(strokes, Workcell())
t0 = time.time()
res = optimise_placement(strokes, dmap)
t_total = time.time() - t0
cell = res["cell"]
n_targets = default.n_points
ik_time_per_point = res["verify_seconds"] / max(res["n_verified"], 1) / n_targets
place = {"drawing_width_mm": 300, "default_origin_mm": list(Workcell().paper_origin_mm),
         "default_min_dexterity": default.min_dexterity, "default_unreachable": default.n_unreachable,
         "optimised_origin_mm": [float(v) for v in cell.paper_origin_mm], "optimised_yaw_deg": cell.yaw_deg,
         "optimised_min_dexterity": res["report"].min_dexterity,
         "optimised_unreachable": res["report"].n_unreachable,
         "candidates_scored": res["n_candidates"], "surrogate_seconds": res["surrogate_seconds"],
         "total_seconds": t_total,
         "brute_force_ik_estimate_s": res["n_candidates"] * n_targets * ik_time_per_point}
print(json.dumps(place, indent=1))
(ROOT / "results" / "dexterity_map.json").write_text(json.dumps({"held_out": ev, "placement_demo": place}, indent=2))

# ---------------------------------------------------------------- figure
BLUE, ORANGE, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlecolor": INK})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=BG)
for a in ax:
    a.set_facecolor(BG)
z_slice = cell.paper_origin_mm[2]
gx, gy = np.meshgrid(np.linspace(-200, 1800, 220), np.linspace(-1600, 1600, 300))
G = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, z_slice)]
gr, gd = dmap.predict(G)
img = np.where(gr > 0.5, gd, np.nan).reshape(gx.shape)
im = ax[0].imshow(img, origin="lower", extent=(-200, 1800, -1600, 1600), cmap="Blues", vmin=0, vmax=0.3, aspect="equal")
fig.colorbar(im, ax=ax[0], fraction=0.035, label="predicted dexterity (0 = singular, 1 = isotropic)")
ax[0].plot(0, 0, marker="s", color=INK, ms=7)
ax[0].text(40, 40, "robot base", fontsize=8, color=INK)


def outline(cellx, color, label):
    xy = np.vstack(strokes)
    lo, hi = xy.min(0), xy.max(0)
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]]])
    W = np.array([np.asarray(cellx.paper_origin_mm) + cellx.R_paper @ np.r_[b, 0] for b in box])
    ax[0].plot(W[:, 0], W[:, 1], color=color, lw=2, label=label)


outline(Workcell(), ORANGE, f"default placement (z = {Workcell().paper_origin_mm[2]:.0f} mm)")
outline(cell, INK, f"optimised placement (z = {z_slice:.0f} mm)")
ax[0].legend(frameon=False, fontsize=8, loc="lower left")
ax[0].set(title=f"Learned dexterity map, tool down (z = {z_slice:.0f} mm)", xlabel="x (mm)", ylabel="y (mm)")

lab = ["default", "optimised"]
vals = [default.min_dexterity, res["report"].min_dexterity]
bars = ax[1].bar(lab, vals, color=[ORANGE, BLUE], width=0.55, edgecolor=BG, lw=2)
for b_, v, u in zip(bars, vals, [default.n_unreachable, res["report"].n_unreachable]):
    ax[1].text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}\n{u} unreachable", ha="center", va="bottom",
               fontsize=8, color=INK)
ax[1].set(title="Worst-case dexterity along a 300 mm drawing", ylabel="min dexterity (inverse condition no.)",
          ylim=(0, max(vals) * 1.35))
ax[1].grid(axis="y", color=GRID)

tt = [place["brute_force_ik_estimate_s"], place["total_seconds"]]
bars = ax[2].bar(["brute-force IK search", "learned map + 5 IK checks"], tt, color=[ORANGE, BLUE],
                 width=0.55, edgecolor=BG, lw=2)
for b_, v in zip(bars, tt):
    ax[2].text(b_.get_x() + b_.get_width() / 2, v, f"{v:,.0f} s" if v > 10 else f"{v:.1f} s",
               ha="center", va="bottom", fontsize=9, color=INK)
ax[2].set(title=f"Time to search {res['n_candidates']:,} placements", ylabel="seconds (log)", yscale="log",
          ylim=(0.5, tt[0] * 5))
ax[2].grid(axis="y", color=GRID)
fig.tight_layout()
fig.savefig(ROOT / "results" / "dexterity_map.png", dpi=140)
