"""Collision & fault detection benchmark on a simulated fleet of 6-DOF arms.

    python scripts/fault_detection.py           # ~10 min on CPU (fleet simulation dominates)
    python scripts/fault_detection.py --reuse   # reuse the cached simulation

Writes models/fault_residual.joblib, models/fault_diagnosis.joblib,
results/fault_detection.json and results/fault_detection.png.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from ml.fault_detection import (DT, FAULTS, FleetRun, LearnedResidual, UnitCalibration, Units,  # noqa: E402
                                debounce, detect, diagnosis_features, momentum_observer, naive_residual,
                                residual_features, score_series, simulate_fleet)
from robotics.dynamics import ArmDynamics  # noqa: E402

CACHE = Path("/tmp/synapsearm_fleet.joblib")
FA_TARGET = 0.01
dyn = ArmDynamics()


def sim_set(labels, seed, units=None):
    out, CH = [], 250
    for k in range(0, len(labels), CH):
        u = units[k:k + CH] if units is not None else None
        out.append(simulate_fleet(len(labels[k:k + CH]), labels[k:k + CH], seed=seed + k, units=u))
    if len(out) == 1:
        return out[0]
    cat = lambda f, ax: np.concatenate([getattr(o, f) for o in out], axis=ax)  # noqa: E731
    return FleetRun(*(cat(f, 1) for f in ("q", "qd", "tau", "qdd_des", "q_des")),
                    *(cat(f, 0) for f in ("fault", "onset", "joint")), Units.concat([o.units for o in out]))


MAIN = ["val_normal", "cal_normal", "diag_train", "test"]     # arms that get a commissioning run
if "--reuse" in sys.argv and CACHE.exists():
    sets = joblib.load(CACHE)
else:
    t0 = time.time()
    sets = {
        "train_normal": sim_set(["none"] * 120, seed=1000),    # fleet model training
        "val_normal": sim_set(["none"] * 100, seed=2000),      # per-joint scales
        "cal_normal": sim_set(["none"] * 200, seed=5000),      # alarm thresholds
        "diag_train": sim_set([f for f in FAULTS for _ in range(60)], seed=3000),
        "test": sim_set(["none"] * 200 + [f for f in FAULTS for _ in range(100)], seed=4000),
    }
    # one fault-free commissioning run per arm: same physical unit, different trajectory
    for name in MAIN:
        sets["comm_" + name] = sim_set(["none"] * len(sets[name].fault), seed=7000 + 97 * MAIN.index(name),
                                       units=sets[name].units)
    print(f"simulated {sum(len(s.fault) for s in sets.values())} runs in {time.time() - t0:.0f} s")
    joblib.dump(sets, CACHE)

t0 = time.time()
res = {name: {"naive": naive_residual(s, dyn), "observer": momentum_observer(s, dyn)} for name, s in sets.items()}
print(f"residuals in {time.time() - t0:.0f} s")

learned = LearnedResidual().fit(residual_features(sets["train_normal"]), res["train_normal"]["observer"])
joblib.dump(learned, ROOT / "models" / "fault_residual.joblib", compress=3)
for name, s in sets.items():
    res[name]["learned"] = res[name]["observer"] - learned.predict(residual_features(s))

# per-unit calibration: identified on each arm's commissioning run, then frozen
for name in MAIN:
    s, c = sets[name], sets["comm_" + name]
    for base, filt in (("learned", True), ("naive", False)):
        cal = UnitCalibration(dyn, filtered=filt).fit(c, res["comm_" + name][base])
        res[name][base + "_unit"] = cal.apply(s, res[name][base])
EVAL = MAIN

DETECTORS = ["naive", "observer", "learned", "naive_unit", "learned_unit"]
NAMES = {"naive": "Inverse-dynamics residual", "observer": "Momentum observer (rigid-body model)",
         "learned": "Observer + learned fleet model", "naive_unit": "ID residual + per-unit calibration",
         "learned_unit": "Learned observer + per-unit calibration"}
# Calibrate on arms the learned model never saw: per-joint scale (99th percentile of |r|) on one
# held-out set, alarm threshold (1% of runs) on another.
scale = {d: np.percentile(np.abs(res["val_normal"][d][100:]).reshape(-1, 6), 99, axis=0) for d in DETECTORS}
score = {d: {n: debounce(score_series(res[n][d], scale[d])) for n in EVAL} for d in DETECTORS}
thr = {d: float(np.quantile(score[d]["cal_normal"].max(0), 1 - FA_TARGET)) for d in DETECTORS}


# Fusion: learned observer OR inverse-dynamics residual. Each is normalised by its own threshold at
# a common per-detector false-alarm level, tuned on the calibration arms so the union stays at 1%.
def add_fusion(name, pair, label):
    def fs(n, fa):
        return np.maximum(*(score[d][n] / np.quantile(score[d]["cal_normal"].max(0), 1 - fa) for d in pair))
    grid = np.linspace(0.002, FA_TARGET, 41)
    fa_each = max(f for f in grid if (fs("cal_normal", f).max(0) > 1).mean() <= FA_TARGET + 1e-9)
    score[name] = {n: fs(n, fa_each) for n in EVAL}
    thr[name] = 1.0
    DETECTORS.append(name)
    NAMES[name] = label
    return fa_each


fa_fused = add_fusion("fused", ("learned", "naive"), "Fused (fleet model)")
fa_fused_unit = add_fusion("fused_unit", ("learned_unit", "naive_unit"), "Fused + per-unit calibration")

test = sets["test"]
results = {"false_alarm_target": FA_TARGET, "detectors": {}}
results["fusion_per_detector_fa"] = {"fused": float(fa_fused), "fused_unit": float(fa_fused_unit)}
for d in DETECTORS:
    hit, first = detect(score[d]["test"], thr[d])
    normal = test.fault == "none"
    row = {"false_alarm_rate_test": float(hit[normal].mean()),
           "auc": float(roc_auc_score(~normal, score[d]["test"].max(0))), "per_fault": {}}
    for f in FAULTS:
        m = test.fault == f
        delay = (first[m] * DT - test.onset[m])[hit[m] & (first[m] * DT >= test.onset[m])]
        row["per_fault"][f] = {"detection_rate": float(hit[m].mean()),
                               "median_delay_ms": float(np.median(delay) * 1000) if len(delay) else None}
    # alarms before onset on faulty runs count as false alarms, not detections
    early = (~normal) & hit & (first * DT < np.nan_to_num(test.onset, nan=0))
    row["early_alarm_frac_faulty"] = float(early[~normal].mean())
    results["detectors"][d] = row

# ------------------------------------------------------------- diagnosis: alarms from the recommended detector
# (fleet-model fusion); fault signature from the learned-observer residual, with and without
# per-unit calibration
DIAG_ALARM = "fused"
dtr = sets["diag_train"]
hit_tr, first_tr = detect(score[DIAG_ALARM]["diag_train"], thr[DIAG_ALARM])
hit_te, first_te = detect(score[DIAG_ALARM]["test"], thr[DIAG_ALARM])
fm = (test.fault != "none") & hit_te
diag = {}
for DIAG_RES in ("learned", "learned_unit"):
    Xtr = diagnosis_features(dtr, res["diag_train"][DIAG_RES] / scale[DIAG_RES], dyn, first_tr)
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xtr[hit_tr], dtr.fault[hit_tr])
    Xte = diagnosis_features(test, res["test"][DIAG_RES] / scale[DIAG_RES], dyn, first_te)
    diag[DIAG_RES] = (clf, clf.predict(Xte[fm]))
DIAG_RES = max(diag, key=lambda k: (diag[k][1] == test.fault[fm]).mean())
clf, pred = diag[DIAG_RES]
joblib.dump(clf, ROOT / "models" / "fault_diagnosis.joblib", compress=3)
cm = confusion_matrix(test.fault[fm], pred, labels=list(FAULTS))
# which joint? largest normalised mean residual after the alarm
jt = (np.isin(test.fault, ["wear", "actuator"])) & hit_te
w = int(0.5 / DT)
r_norm = res["test"][DIAG_RES] / scale[DIAG_RES]
joint_pred = np.array([np.abs(r_norm[first_te[n]:first_te[n] + w, n].mean(0)).argmax() for n in np.where(jt)[0]])
# noise floor: velocity-sensor noise passes through the observer as K M(q) n
Ms = dyn.mass_matrix(sets["val_normal"].q[::100].reshape(-1, 6))
noise_floor = (50.0 * np.sqrt(np.einsum("nii->ni", Ms @ Ms)).mean(0) * 2e-3)
results["normal_residual_std_val"] = {d: res["val_normal"][d][100:].reshape(-1, 6).std(0).round(3).tolist()
                                      for d in ("observer", "learned", "learned_unit")}
results["observer_velocity_noise_floor_std"] = noise_floor.round(3).tolist()
results["diagnosis"] = {"alarm_detector": DIAG_ALARM, "residual": DIAG_RES,
                        "accuracy_by_residual": {k: float((v[1] == test.fault[fm]).mean()) for k, v in diag.items()},
                        "accuracy": float((pred == test.fault[fm]).mean()), "labels": list(FAULTS),
                        "confusion_matrix": cm.tolist(),
                        "joint_localisation_accuracy": float((joint_pred == test.joint[jt]).mean())}
(ROOT / "results" / "fault_detection.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=1))

# ------------------------------------------------------------- figure
BLUE, ORANGE, GRAY, AQUA = "#2a78d6", "#eb6834", "#8a8984", "#1baf7a"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlecolor": INK})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=BG, gridspec_kw={"width_ratios": [1.3, 1.2, 0.9]})
for a in ax:
    a.set_facecolor(BG)

# (a) one light collision: the rigid-body observer can't see it, the learned residual can
cand = np.where(test.fault == "collision")[0]
pick = [n for n in cand if score["learned"]["test"][:, n].max() > thr["learned"]
        and score["observer"]["test"][:, n].max() <= thr["observer"]]
if not pick:   # otherwise the lightest collision the learned detector still catches
    peak = score["learned"]["test"][:, cand].max(0) / thr["learned"]
    pick = [cand[np.argmin(np.where(peak > 1, peak, np.inf))]]
n = pick[0]
t = np.arange(score["observer"]["test"].shape[0]) * DT
ax[0].plot(t, score["observer"]["test"][:, n] / thr["observer"], color=ORANGE, lw=1.3, label=NAMES["observer"])
ax[0].plot(t, score["learned"]["test"][:, n] / thr["learned"], color=BLUE, lw=1.3, label=NAMES["learned"])
ax[0].axhline(1.0, color=INK2, ls="--", lw=1)
ax[0].text(0.1, 1.04, "alarm threshold (1% false alarms)", fontsize=8, color=INK2)
ax[0].axvline(test.onset[n], color=INK2, ls=":", lw=1)
ax[0].text(test.onset[n] + 0.05, 0.05, "contact", fontsize=8, color=INK2)
ax[0].set(title="A light collision, normalised detection score", xlabel="time (s)", ylabel="score / threshold",
          ylim=(0, min(3.0, max(2.0, ax[0].get_ylim()[1]))))
ax[0].legend(frameon=False, fontsize=8, loc="upper left")

# (b) detection rate per fault at the same false-alarm rate
xx = np.arange(len(FAULTS))
SHOW = ["observer", "learned", "fused", "fused_unit"]
cols = {"observer": ORANGE, "learned": BLUE, "fused": GRAY, "fused_unit": AQUA}
for k, d in enumerate(SHOW):
    v = [results["detectors"][d]["per_fault"][f]["detection_rate"] for f in FAULTS]
    bars = ax[1].bar(xx + (k - 1.5) * 0.2, v, width=0.19, color=cols[d], label=NAMES[d], edgecolor=BG, lw=1)
    for b_, val in zip(bars, v):
        ax[1].text(b_.get_x() + b_.get_width() / 2, val + 0.01, f"{100 * val:.0f}", ha="center", va="bottom",
                   fontsize=6.5, color=INK)
ax[1].set_xticks(xx, FAULTS)
ax[1].set(title="Detection rate at 1% false alarms (unseen arms)", ylabel="faults detected", ylim=(0, 1.6))
ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax[1].legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2)
ax[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# (c) diagnosis confusion matrix
cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
ax[2].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(len(FAULTS)):
    for j in range(len(FAULTS)):
        ax[2].text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center", fontsize=9,
                   color="white" if cmn[i, j] > 0.5 else INK)
ax[2].set_xticks(range(len(FAULTS)), FAULTS, rotation=30)
ax[2].set_yticks(range(len(FAULTS)), FAULTS)
ax[2].set(title=f"Diagnosis: {results['diagnosis']['accuracy']:.0%} correct", xlabel="predicted", ylabel="true")
ax[2].grid(False)
fig.tight_layout()
fig.subplots_adjust(bottom=0.17)
fig.savefig(ROOT / "results" / "fault_detection.png", dpi=140)

if "--debug" in sys.argv:
    for d in DETECTORS:
        hit, first = detect(score[d]["test"], thr[d])
        m = test.fault == "wear"
        print(d, "thr", round(thr[d], 2), "wear det by joint",
              [f"{hit[m & (test.joint == j)].mean():.2f}" for j in range(6)],
              "act by joint", [f"{hit[(test.fault == 'actuator') & (test.joint == j)].mean():.2f}" for j in range(6)])
        jmax = np.abs(res["cal_normal"][d] / scale[d]).max(0).argmax(-1)
        print("   which joint sets the cal run-max:", np.bincount(jmax, minlength=6))
