"""Train the neural IK model, benchmark it, and save model + results.

    python scripts/train_neural_ik.py --samples 150000
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.neural_ik import NeuralIK, benchmark  # noqa: E402
from robotics.kinematics import DHRobot  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--samples", type=int, default=150_000)
p.add_argument("--max-iter", type=int, default=200)
p.add_argument("--targets", type=int, default=500)
args = p.parse_args()

(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)

nik = NeuralIK(DHRobot()).fit(n_samples=args.samples, max_iter=args.max_iter)
nik.save(ROOT / "models" / "neural_ik.joblib")

res = benchmark(nik, n_targets=args.targets)
(ROOT / "results" / "neural_ik_benchmark.json").write_text(json.dumps(res, indent=2))

print(f"\n{'Strategy':30s} {'success':>8s} {'median err (mm)':>16s} {'iters':>7s} {'ms/solve':>9s}")
for k, v in res.items():
    print(f"{k:30s} {v['success_rate']:8.1%} {v['median_pos_err_mm']:16.4f} {v['mean_iters']:7.1f} {v['ms_per_solve']:9.3f}")
