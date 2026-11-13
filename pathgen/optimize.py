"""Path optimisation: simplify strokes within a tolerance, then order them to minimise pen-up travel.

Stroke ordering is a travelling-salesman problem where each stroke can be drawn in either
direction (a generalised TSP). We use greedy nearest-neighbour construction followed by 2-opt
local search over the stroke sequence (segment reversal also flips each stroke's direction).
"""
from __future__ import annotations

import numpy as np


def rdp(points: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker: drop points while keeping deviation <= tol (iterative)."""
    pts = np.asarray(points, float)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        seg = b - a
        L = np.linalg.norm(seg)
        rel = pts[i + 1:j] - a
        if L < 1e-12:
            d = np.linalg.norm(rel, axis=1)
        else:
            d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return pts[keep]


def simplify(strokes, tol_mm=0.1):
    return [rdp(s, tol_mm) for s in strokes]


def pen_up_travel(strokes, start=(0.0, 0.0)) -> float:
    pos, total = np.asarray(start, float), 0.0
    for s in strokes:
        total += np.linalg.norm(s[0] - pos)
        pos = s[-1]
    return float(total)


def draw_length(strokes) -> float:
    return float(sum(np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in strokes))


def order_strokes(strokes, start=(0.0, 0.0), two_opt=True, max_rounds=50):
    """Greedy nearest-neighbour (either end) + 2-opt. Returns re-ordered, possibly reversed strokes."""
    n = len(strokes)
    if n <= 1:
        return list(strokes)
    heads = np.array([s[0] for s in strokes])
    tails = np.array([s[-1] for s in strokes])
    remaining = np.ones(n, bool)
    pos = np.asarray(start, float)
    order = []  # (index, reversed)
    for _ in range(n):
        dh = np.where(remaining, np.linalg.norm(heads - pos, axis=1), np.inf)
        dt = np.where(remaining, np.linalg.norm(tails - pos, axis=1), np.inf)
        i_h, i_t = int(np.argmin(dh)), int(np.argmin(dt))
        if dh[i_h] <= dt[i_t]:
            order.append((i_h, False)); pos = tails[i_h]; remaining[i_h] = False  # noqa: E702
        else:
            order.append((i_t, True)); pos = heads[i_t]; remaining[i_t] = False  # noqa: E702

    def ends(o):
        i, r = o
        return (tails[i], heads[i]) if r else (heads[i], tails[i])

    if two_opt and n > 2:
        start = np.asarray(start, float)
        for _ in range(max_rounds):
            improved = False
            for a in range(0, n - 1):
                prev_end = start if a == 0 else ends(order[a - 1])[1]
                for b in range(a + 1, n):
                    nxt_start = ends(order[b + 1])[0] if b + 1 < n else None
                    s_a, e_b = ends(order[a])[0], ends(order[b])[1]
                    old = np.linalg.norm(s_a - prev_end) + (np.linalg.norm(nxt_start - e_b) if nxt_start is not None else 0)
                    # reversing order[a..b] (and each stroke's direction) swaps which ends connect
                    new = np.linalg.norm(e_b - prev_end) + (np.linalg.norm(nxt_start - s_a) if nxt_start is not None else 0)
                    if new < old - 1e-9:
                        order[a:b + 1] = [(i, not r) for i, r in reversed(order[a:b + 1])]
                        improved = True
            if not improved:
                break
    return [strokes[i][::-1] if r else strokes[i] for i, r in order]
