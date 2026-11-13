"""End-to-end: request -> strokes -> simplify -> order -> reachability -> RAPID.

Requests can be plain commands, parsed by a small grammar:
    "draw a 60 mm star"            "draw a hexagon 80mm"
    "write 'Hello robot' 25 mm"    "write SynapseArm"
or an image path (sketch or photo).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pathgen import optimize, sources
from pathgen.rapid import rapid_stats, to_rapid
from pathgen.reach import Workcell, check_reach

SHAPES = ("circle", "star", "heart", "spiral", "hexagon", "pentagon", "square", "triangle", "octagon", "polygon")


def parse_command(cmd: str) -> dict:
    """Small grammar for drawing requests. Returns {'kind': 'text'|'shape', ...}."""
    c = cmd.strip()
    size = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm)\b", c, re.I)
    size_mm = None if not size else float(size.group(1)) * (10 if size.group(2).lower() == "cm" else 1)
    m = re.match(r"^\s*(write|text|print)\s+(.+)$", c, re.I)
    if m:
        body = m.group(2)
        q = re.search(r"[\"'“‘](.+?)[\"'”’]", body)
        text = q.group(1) if q else re.sub(r"\s*\d+(?:\.\d+)?\s*(mm|cm)\b.*$", "", body, flags=re.I).strip()
        return {"kind": "text", "text": text, "height_mm": size_mm or 30.0}
    for s in SHAPES:
        if re.search(rf"\b{s}s?\b", c, re.I):
            n = re.search(r"(\d+)[- ]?(?:point|pointed|sided|sides)", c, re.I)
            return {"kind": "shape", "shape": s, "size_mm": size_mm or 50.0,
                    "n": int(n.group(1)) if n else None}
    raise ValueError(f"Couldn't understand {cmd!r}. Try 'draw a 60 mm star' or \"write 'hi' 25 mm\".")


@dataclass
class Job:
    strokes: list
    raw_strokes: list
    rapid: str
    stats: dict = field(default_factory=dict)


def build(request: str | None = None, image=None, image_mode="lines", width_mm=120.0, tol_mm=0.1,
          cell: Workcell | None = None, check=True, name="SynapseDraw") -> Job:
    if image is not None:
        raw = sources.image_strokes(image, width_mm=width_mm, mode=image_mode)
        desc = f"image: {image}" if isinstance(image, str) else "image"
    else:
        cmd = parse_command(request)
        if cmd["kind"] == "text":
            raw = sources.text_strokes(cmd["text"], cmd["height_mm"])
        else:
            kw = {}
            if cmd["n"]:
                kw = {"points": cmd["n"]} if cmd["shape"] == "star" else {"sides": cmd["n"]}
            raw = sources.shape_strokes(cmd["shape"], cmd["size_mm"], **kw)
        desc = f"request: {request}"
    raw = sources.normalise(raw, margin_mm=10.0)   # keep the drawing 10 mm in from the paper corner
    simp = optimize.simplify(raw, tol_mm)
    ordered = optimize.order_strokes(simp)
    stats = {
        "strokes": len(raw),
        "points_raw": int(sum(len(s) for s in raw)),
        "points_simplified": int(sum(len(s) for s in ordered)),
        "draw_length_mm": optimize.draw_length(ordered),
        "pen_up_travel_naive_mm": optimize.pen_up_travel(simp),
        "pen_up_travel_greedy_mm": optimize.pen_up_travel(optimize.order_strokes(simp, two_opt=False)),
        "pen_up_travel_optimised_mm": optimize.pen_up_travel(ordered),
    }
    cell = cell or Workcell()
    if check:
        rep = check_reach(ordered, cell)
        stats["reachability"] = rep.as_dict()
    code = to_rapid(ordered, name=name, pen_length_mm=cell.pen_length_mm, paper_origin_mm=cell.paper_origin_mm,
                    comment=desc)
    stats["rapid"] = rapid_stats(code)
    return Job(ordered, raw, code, stats)
