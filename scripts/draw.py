"""Turn a request or a sketch into a ready-to-load ABB RAPID module.

    python scripts/draw.py "draw a 60 mm star"
    python scripts/draw.py "write 'Hello robot' 25 mm"
    python scripts/draw.py --image examples/sketch.png --width 160
    python scripts/draw.py --demo          # regenerates results/pathgen_demo.png + examples/*.mod

Writes <out>.mod (RAPID), <out>.png (preview) and <out>.json (stats).
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

from pathgen import optimize  # noqa: E402
from pathgen.pipeline import build  # noqa: E402

BLUE, ORANGE, GRAY, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#b5b4ad", "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"


def style(ax, title):
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_title(title, fontweight="bold", color=INK, fontsize=10)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


def plot_plan(ax, strokes, title, travel_color=ORANGE):
    pos = np.zeros(2)
    for s in strokes:
        ax.plot([pos[0], s[0, 0]], [pos[1], s[0, 1]], color=travel_color, lw=0.8, ls="--", alpha=0.8)
        ax.plot(s[:, 0], s[:, 1], color=BLUE, lw=1.4, solid_capstyle="round")
        pos = s[-1]
    ax.plot([], [], color=BLUE, lw=1.4, label="pen down (draw)")
    ax.plot([], [], color=travel_color, lw=0.8, ls="--", label=f"pen up: {optimize.pen_up_travel(strokes) / 1000:.2f} m")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, title)


def save(job, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".mod").write_text(job.rapid)
    out.with_suffix(".json").write_text(json.dumps(job.stats, indent=2))
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    plot_plan(ax, job.strokes, f"{job.stats['strokes']} strokes, {job.stats['points_simplified']} targets")
    ax.set_xlabel("x on paper (mm)", color=INK2)
    ax.set_ylabel("y (mm)", color=INK2)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)


def demo():
    import cv2
    sketch = ROOT / "examples" / "sketch.png"
    job = build(image=str(sketch), width_mm=160)
    save(job, ROOT / "examples" / "sketch_drawing")
    text = build("write 'SynapseArm' 22 mm")
    save(text, ROOT / "examples" / "text_drawing")
    star = build("draw a 7-pointed star 60 mm")
    save(star, ROOT / "examples" / "star_drawing")

    simp = optimize.simplify(job.raw_strokes, 0.1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=BG)
    ax[0].imshow(cv2.imread(str(sketch), cv2.IMREAD_GRAYSCALE), cmap="gray")
    ax[0].set_xticks([]), ax[0].set_yticks([])
    style(ax[0], "Input: hand-drawn sketch (PNG)")
    plot_plan(ax[1], simp, "Vectorised, traced order", travel_color=GRAY)
    plot_plan(ax[2], job.strokes, "Optimised order (greedy + 2-opt)")
    s = job.stats
    ax[2].text(0.0, -0.13, f"{s['points_raw']} → {s['points_simplified']} targets (≤0.1 mm deviation) · "
               f"{s['reachability']['unreachable']} unreachable · max joint step "
               f"{s['reachability']['max_joint_step_deg']:.1f}° · {s['rapid']['MoveL']} MoveL",
               transform=ax[2].transAxes, fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "pathgen_demo.png", dpi=140)
    summary = {"sketch": job.stats, "text": text.stats, "star": star.stats}
    (ROOT / "results" / "pathgen_demo.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("request", nargs="?")
    p.add_argument("--image")
    p.add_argument("--mode", default="lines", choices=["lines", "edges"], help="lines: sketch, edges: photo")
    p.add_argument("--width", type=float, default=120.0, help="drawing width for images (mm)")
    p.add_argument("--tol", type=float, default=0.1, help="simplification tolerance (mm)")
    p.add_argument("--out", default="results/drawing")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()
    if a.demo:
        demo()
    else:
        job = build(a.request, image=a.image, image_mode=a.mode, width_mm=a.width, tol_mm=a.tol)
        save(job, ROOT / a.out)
        print(json.dumps(job.stats, indent=1))
        print(f"\nRAPID module -> {a.out}.mod")
