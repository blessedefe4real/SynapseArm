"""Generate examples/sketch.png: a wobbly, hand-drawn-looking doodle of a robot arm and a sun."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)
img = np.full((600, 800), 255, np.uint8)


def wobbly(pts, th=5):
    pts = np.asarray(pts, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    t = np.linspace(0, d[-1], max(int(d[-1] / 3), 2))
    P = np.c_[np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])]
    P += np.cumsum(rng.normal(0, 0.35, P.shape), 0) * 0.5      # low-frequency hand wobble
    cv2.polylines(img, [P.astype(np.int32)], False, 0, th, cv2.LINE_AA)


def arc(c, r, n=80, a0=0, a1=2 * np.pi):
    a = np.linspace(a0, a1, n)
    return np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]


wobbly([[120, 540], [300, 540]]); wobbly([[150, 540], [170, 480], [250, 480], [270, 540]])  # noqa: E702
wobbly(arc((210, 470), 28))
wobbly([[200, 445], [320, 280]]); wobbly([[222, 455], [338, 292]])  # noqa: E702
wobbly(arc((330, 285), 22))
wobbly([[340, 268], [480, 220]]); wobbly([[350, 300], [488, 245]])  # noqa: E702
wobbly(arc((490, 232), 16))
wobbly([[500, 222], [545, 200], [560, 215]]); wobbly([[502, 245], [548, 262], [560, 248]])  # noqa: E702
wobbly(arc((660, 120), 55))
for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
    wobbly([[660 + 70 * np.cos(a), 120 + 70 * np.sin(a)], [660 + 100 * np.cos(a), 120 + 100 * np.sin(a)]], 4)
wobbly(arc((640, 105), 5, 20)); wobbly(arc((680, 105), 5, 20))  # noqa: E702
wobbly(arc((660, 125), 28, 40, 0.25 * np.pi, 0.75 * np.pi))
wobbly(np.c_[np.linspace(60, 760, 60), 560 + 6 * np.sin(np.linspace(0, 18, 60))], 4)
(ROOT / "examples").mkdir(exist_ok=True)
cv2.imwrite(str(ROOT / "examples" / "sketch.png"), img)
