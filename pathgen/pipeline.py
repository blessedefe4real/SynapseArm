"""End-to-end: request -> strokes -> simplify -> order -> reachability -> RAPID.

Requests are free text. With an Anthropic API key, Claude plans the layout (pathgen.llm);
otherwise a small grammar handles one or more clauses:
    "draw a 60 mm star"            "write 'Hello robot' 25 mm and a small star"
    "draw a hexagon 8 cm, then a circle"
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
          cell: Workcell | None = None, check=True, name="SynapseDraw", auto_place=False,
          use_llm: bool | None = None) -> Job:
    plan, planner = None, "image"
    if image is not None:
        raw = sources.image_strokes(image, width_mm=width_mm, mode=image_mode)
        desc = f"image: {image}" if isinstance(image, str) else "image"
    else:
        from pathgen.llm import plan_request, plan_to_strokes
        plan, planner = plan_request(request, use_llm=use_llm)
        raw = plan_to_strokes(plan)
        desc = f"request: {request}"
    raw = sources.normalise(raw, margin_mm=10.0)   # keep the drawing 10 mm in from the paper corner
    simp = optimize.simplify(raw, tol_mm)
    ordered = optimize.order_strokes(simp)
    stats = {
        "planner": planner,
        **({"plan": plan} if plan else {}),
        "strokes": len(raw),
        "points_raw": int(sum(len(s) for s in raw)),
        "points_simplified": int(sum(len(s) for s in ordered)),
        "draw_length_mm": optimize.draw_length(ordered),
        "pen_up_travel_naive_mm": optimize.pen_up_travel(simp),
        "pen_up_travel_greedy_mm": optimize.pen_up_travel(optimize.order_strokes(simp, two_opt=False)),
        "pen_up_travel_optimised_mm": optimize.pen_up_travel(ordered),
    }
    cell = cell or Workcell()
    if auto_place:
        from ml.dexterity import DexterityMap, optimise_placement
        res = optimise_placement(ordered, DexterityMap.load(), verbose=False)
        if res["cell"] is not None:
            cell = res["cell"]
            stats["placement"] = {"paper_origin_mm": [round(v, 1) for v in cell.paper_origin_mm],
                                  "yaw_deg": round(cell.yaw_deg, 1),
                                  "candidates_scored": res["n_candidates"],
                                  "surrogate_seconds": round(res["surrogate_seconds"], 2)}
    if check:
        rep = check_reach(ordered, cell)
        stats["reachability"] = rep.as_dict()
    code = to_rapid(ordered, name=name, pen_length_mm=cell.pen_length_mm, paper_origin_mm=cell.paper_origin_mm,
                    yaw_deg=cell.yaw_deg,
                    comment=desc)
    stats["rapid"] = rapid_stats(code)
    return Job(ordered, raw, code, stats)
