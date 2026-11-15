"""Full 6-DOF arm: learned feed-forward from one excitation run, plus ES-tuned servo gains.

    python scripts/arm_learning.py      # ~15 min on CPU

Writes results/arm_learning.json and results/arm_learning.png.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from ml.arm_learning import (WEIGHT_JOINT, ArmFeedForward, default_log_gains, simulate_gains,  # noqa: E402
                             tracking_rms, train_gains_es, true_unmodelled_torque)
from ml.fault_detection import DT, Units, random_moves, simulate_fleet  # noqa: E402
from robotics.dynamics import ArmDynamics  # noqa: E402

dyn = ArmDynamics()
robot = dyn.robot


def learn_feedforward(units, seed, use_gp=True):
    """One 10 s excitation run per arm (default gains, no feed-forward), then fit."""
    run = simulate_fleet(len(units.payload_dm), seed=seed, units=units, trajectory="excitation", t_run=10.0)
    return ArmFeedForward(dyn, use_gp=use_gp).fit(run)


def tool_error_mm(run, stride=20):
    """RMS and max tool-flange position error (mm) from forward kinematics, per run."""
    Q, Qd = run.q[::stride], run.q_des[::stride]
    d = np.empty(Q.shape[:2])
    for t in range(Q.shape[0]):
        for n in range(Q.shape[1]):
            d[t, n] = np.linalg.norm(robot.fk(Q[t, n])[:3, 3] - robot.fk(Qd[t, n])[:3, 3])
    return np.sqrt((d ** 2).mean(0)), d.max(0)


# ---------------------------------------------------------------- ES gain tuning on training arms
t0 = time.time()
train_units = Units.sample(6, np.random.default_rng(101), dyn)
ff_train = learn_feedforward(train_units, seed=102)
print("Tuning servo gains with evolution strategies (learned feed-forward in the loop)...")
theta, chatter_ref, hist = train_gains_es(train_units, ff_model=ff_train, generations=25)
print(f"  done in {time.time() - t0:.0f} s")

# ---------------------------------------------------------------- held-out evaluation
test_units = Units.sample(12, np.random.default_rng(202), dyn)
t1 = time.time()
ff_gp = learn_feedforward(test_units, seed=203, use_gp=True)
ff_lin = learn_feedforward(test_units, seed=203, use_gp=False)
print(f"learned feed-forward for 12 held-out arms in {time.time() - t1:.0f} s")
ref = random_moves(12, np.random.default_rng(204), robot)                  # unseen 6 s trajectories
kp_es, kd_es = np.exp(theta[:6]), np.exp(theta[6:])
variants = {
    "Rigid-body model only": dict(),
    "+ learned feed-forward (physics basis)": dict(ff=ff_lin.predict(ref[0], ref[1])),
    "+ learned feed-forward (physics + GP)": dict(ff=ff_gp.predict(ref[0], ref[1])),
    "+ learned feed-forward + ES gains": dict(ff=ff_gp.predict(ref[0], ref[1]), kp=kp_es, kd=kd_es),
    "Perfect model (oracle feed-forward)": dict(ff=true_unmodelled_torque(test_units, dyn, ref[0], ref[1])),
}
results = {"es_gains": {"kp": kp_es.round(1).tolist(), "kd": kd_es.round(2).tolist(),
                        "default_kp": 400.0, "default_kd": 40.0},
           "es_history": hist.tolist(), "variants": {}}
runs = {}
for name, kw in variants.items():
    run = simulate_fleet(12, seed=205, units=test_units, ref=ref, **kw)
    runs[name] = run
    joint_mrad = tracking_rms(run).mean(0) * 1e3
    tool_rms, tool_max = tool_error_mm(run)
    chatter = float(np.abs(np.diff(run.tau, axis=0) / DT).mean())
    results["variants"][name] = {"joint_rms_mrad": joint_mrad.round(3).tolist(),
                                 "weighted_mrad": float((joint_mrad * WEIGHT_JOINT).sum()),
                                 "tool_rms_mm": float(tool_rms.mean()), "tool_max_mm": float(tool_max.max()),
                                 "torque_chatter": chatter}
    print(f"{name:40s} tool RMS {tool_rms.mean():6.3f} mm  max {tool_max.max():6.2f} mm  "
          f"joints {joint_mrad.round(2)}  chatter {chatter:.0f}")
(ROOT / "results" / "arm_learning.json").write_text(json.dumps(results, indent=2))

# ---------------------------------------------------------------- figure
BLUE, ORANGE, GRAY, AQUA = "#2a78d6", "#eb6834", "#8a8984", "#1baf7a"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlecolor": INK})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=BG, gridspec_kw={"width_ratios": [1.25, 1.1, 1.0]})
for a in ax:
    a.set_facecolor(BG)

names = list(variants)
t = np.arange(ref[0].shape[0]) * DT
n, j = 0, 3
for name, col, lw in [(names[0], ORANGE, 1.2), (names[2], BLUE, 1.4)]:
    ax[0].plot(t, (runs[name].q_des[:, n, j] - runs[name].q[:, n, j]) * 1e3, color=col, lw=lw,
               label=name.replace("+ ", "with "))
ax[0].axhline(0, color=INK2, lw=0.8)
ax[0].set(title="Wrist joint 4 on an unseen trajectory", xlabel="time (s)", ylabel="tracking error (mrad)")
ax[0].legend(frameon=False, fontsize=8, loc="upper right")
ax[0].grid(axis="y", color=GRID)

vals = [results["variants"][k]["tool_rms_mm"] for k in names]
labels = ["rigid-body\nmodel", "+ learned\nphysics", "+ learned\nphysics + GP", "+ ES\ngains", "perfect\nmodel"]
cols = [ORANGE, GRAY, BLUE, AQUA, INK2]
bars = ax[1].bar(range(len(vals)), vals, color=cols, width=0.62, edgecolor=BG, lw=2)
for b_, v in zip(bars, vals):
    ax[1].text(b_.get_x() + b_.get_width() / 2, v * 1.08, f"{v:.3f}" if v < 1 else f"{v:.2f}", ha="center",
               va="bottom", fontsize=8, color=INK)
ax[1].set_xticks(range(len(vals)), labels, fontsize=8.5)
ax[1].set(title="Tool position error, 12 held-out arms", ylabel="RMS tool-flange error (mm, log)", yscale="log",
          ylim=(min(vals) * 0.5, max(vals) * 3))
ax[1].grid(axis="y", color=GRID)

h = np.array(hist)
g = np.arange(1, len(h) + 1)
ax[2].plot(g, h[:, 1], color=BLUE, lw=2, label="tracking (weighted mrad)")
ax[2].set(title="Servo-gain tuning by evolution strategies", xlabel="generation", ylabel="tracking (mrad)")
ax[2].grid(axis="y", color=GRID)
ax[2].legend(frameon=False, fontsize=8, loc="upper right")
ax[2].text(0.02, 0.04, "chatter held at or below the factory gains' level\n"
           f"kp {np.exp(theta[:6]).round(0).astype(int).tolist()}\nkd {np.exp(theta[6:]).round(0).astype(int).tolist()}",
           transform=ax[2].transAxes, fontsize=8, color=INK2)
fig.tight_layout()
fig.savefig(ROOT / "results" / "arm_learning.png", dpi=140)
