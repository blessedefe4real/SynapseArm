"""Turn text, parametric shapes, or a sketch image into pen strokes (lists of (N, 2) polylines, mm).

All outputs are in a paper frame: x to the right, y up, origin at the drawing's bottom-left.
"""
from __future__ import annotations

import numpy as np

Stroke = np.ndarray  # (N, 2) mm


def normalise(strokes: list[Stroke], width_mm=None, height_mm=None, margin_mm=0.0) -> list[Stroke]:
    """Scale uniformly to the requested width or height and move to the origin."""
    pts = np.vstack(strokes)
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-9)
    if width_mm is not None:
        s = width_mm / span[0]
    elif height_mm is not None:
        s = height_mm / span[1]
    else:
        s = 1.0
    return [(st - lo) * s + margin_mm for st in strokes]


# ----------------------------------------------------------------------------- text
def text_strokes(text: str, height_mm=30.0, font="DejaVu Sans", tol_mm=0.2) -> list[Stroke]:
    """Glyph outlines via matplotlib's font engine; Bezier curves flattened to tol_mm."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    tp = TextPath((0, 0), text, size=100, prop=FontProperties(family=font))
    scale = height_mm / 100 * 100 / max(tp.get_extents().height, 1e-9)
    polys = tp.to_polygons(closed_only=False)
    strokes = [np.asarray(p, float) * scale for p in polys if len(p) >= 2]
    return [_resample_curve(s, tol_mm) for s in strokes]


def _resample_curve(s, step):
    """Densify long segments so later simplification controls the error, not the font flattener."""
    out = [s[0]]
    for a, b in zip(s[:-1], s[1:]):
        n = max(int(np.linalg.norm(b - a) / (20 * step)), 1)
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.array(out)


# ----------------------------------------------------------------------------- shapes
def shape_strokes(name: str, size_mm=50.0, sides=6, points=5) -> list[Stroke]:
    """Parametric shapes, sized by their bounding-box width."""
    t = np.linspace(0, 2 * np.pi, 361)
    name = name.lower()
    if name == "circle":
        s = np.c_[np.cos(t), np.sin(t)]
    elif name in ("polygon", "hexagon", "pentagon", "square", "triangle", "octagon"):
        n = {"hexagon": 6, "pentagon": 5, "square": 4, "triangle": 3, "octagon": 8}.get(name, sides)
        a = np.linspace(0, 2 * np.pi, n + 1) + np.pi / 2 - (np.pi / n if n % 2 == 0 else 0)
        s = np.c_[np.cos(a), np.sin(a)]
    elif name == "star":
        a = np.linspace(0, 2 * np.pi, 2 * points + 1) + np.pi / 2
        r = np.where(np.arange(2 * points + 1) % 2 == 0, 1.0, 0.382)
        s = np.c_[r * np.cos(a), r * np.sin(a)]
    elif name == "heart":
        s = np.c_[16 * np.sin(t) ** 3, 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)]
    elif name == "spiral":
        u = np.linspace(0, 6 * np.pi, 1200)
        s = np.c_[u * np.cos(u), u * np.sin(u)]
    else:
        raise ValueError(f"unknown shape {name!r}")
    return normalise([s], width_mm=size_mm)


# ----------------------------------------------------------------------------- images
def image_strokes(img, width_mm=100.0, mode="lines", min_len_px=8) -> list[Stroke]:
    """Vectorise a sketch or photo into single-line strokes.

    mode="lines": dark strokes on light paper (a drawing). Binarise (Otsu), skeletonise to
                  1-px centre-lines, then trace the skeleton graph into polylines.
    mode="edges": photo. Canny edge detection, then the same skeleton tracing.
    img: path or (H, W[, 3]) array.
    """
    import cv2
    from skimage.morphology import skeletonize

    if isinstance(img, (str, bytes)) or hasattr(img, "__fspath__"):
        img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
    elif img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img.astype(np.uint8), (3, 3), 0)
    if mode == "edges":
        mask = cv2.Canny(img, 50, 150) > 0
    else:
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = bw > 0
    skel = skeletonize(mask)
    paths = trace_skeleton(skel)
    paths = [p for p in paths if len(p) >= min_len_px]
    if not paths:
        return []
    # pixel (row, col) -> paper (x right, y up)
    strokes = [np.c_[p[:, 1], skel.shape[0] - p[:, 0]].astype(float) for p in paths]
    return normalise(strokes, width_mm=width_mm)


_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_skeleton(skel: np.ndarray) -> list[np.ndarray]:
    """Walk a 1-px skeleton into polylines. Starts at endpoints/junctions, then mops up loops."""
    H, W = skel.shape
    on = {(int(r), int(c)) for r, c in zip(*np.nonzero(skel))}

    def nbrs(p):
        return [(p[0] + dr, p[1] + dc) for dr, dc in _NB if (p[0] + dr, p[1] + dc) in on]

    degree = {p: len(nbrs(p)) for p in on}
    used_edges: set = set()
    paths = []

    def walk(start, nxt):
        path = [start, nxt]
        used_edges.add(frozenset((start, nxt)))
        prev, cur = start, nxt
        while degree[cur] == 2:
            cand = [q for q in nbrs(cur) if q != prev and frozenset((cur, q)) not in used_edges]
            if not cand:
                break
            prev, cur = cur, cand[0]
            used_edges.add(frozenset((prev, cur)))
            path.append(cur)
        return path

    # 1) from endpoints and junctions
    for p in sorted(on, key=lambda q: degree[q] == 2):
        if degree[p] == 2:
            continue
        for q in nbrs(p):
            if frozenset((p, q)) not in used_edges:
                paths.append(walk(p, q))
    # 2) closed loops with no endpoint
    for p in on:
        for q in nbrs(p):
            if frozenset((p, q)) not in used_edges:
                paths.append(walk(p, q))
    return [np.array(pth) for pth in paths]
