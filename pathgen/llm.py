"""Natural-language drawing requests -> a validated drawing plan.

    plan = plan_request("sign 'Ada' about 3 cm tall with a small star to the right")
    -> [{"kind": "text", "text": "Ada", "height_mm": 30, "x_mm": 0, "y_mm": 0},
        {"kind": "shape", "shape": "star", "size_mm": 15, "x_mm": 75, "y_mm": 8}]

With ANTHROPIC_API_KEY set (and the `anthropic` package installed), Claude turns free-form
requests into this plan via tool use with a strict JSON schema. Everything it returns is then
validated and clamped here, so the robot never receives an unchecked size or position.
Without a key, a rule-based parser handles one or more simple clauses ("a 60 mm star and the
word 'hi'"). The model is configurable with SYNAPSEARM_LLM_MODEL.
"""
from __future__ import annotations

import os
import re

from pathgen import sources
from pathgen.pipeline import SHAPES, parse_command

DEFAULT_MODEL = "claude-haiku-5-5"
MAX_ITEMS = 12
SIZE_RANGE_MM = (3.0, 400.0)
POS_RANGE_MM = (-400.0, 400.0)

PLAN_TOOL = {
    "name": "drawing_plan",
    "description": "Lay out what the robot should draw on paper. Units are millimetres; x to the right, "
                   "y up, origin at the bottom-left of the first item. Omit x_mm/y_mm to place items in a row.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array", "maxItems": MAX_ITEMS,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["text", "shape"]},
                        "text": {"type": "string", "description": "for kind=text"},
                        "height_mm": {"type": "number", "description": "text height"},
                        "shape": {"type": "string", "enum": list(SHAPES)},
                        "size_mm": {"type": "number", "description": "shape width"},
                        "n": {"type": "integer", "description": "star points or polygon sides"},
                        "x_mm": {"type": "number"},
                        "y_mm": {"type": "number"},
                    },
                    "required": ["kind"],
                },
            },
        },
        "required": ["items"],
    },
}

SYSTEM = ("You convert requests for a pen-plotting robot arm into a drawing plan. Only use the "
          "drawing_plan tool. Supported: text (any string) and shapes: " + ", ".join(SHAPES) + ". "
          "Infer reasonable sizes in mm when not given (handwriting-sized text is ~20-30 mm). "
          "If something can't be drawn with these primitives, approximate it with them.")


# ----------------------------------------------------------------------------- validation
def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return min(max(v, lo), hi)


def validate_plan(items) -> list[dict]:
    """Drop anything malformed, clamp sizes and positions, cap the item count."""
    out = []
    for it in list(items)[:MAX_ITEMS]:
        if not isinstance(it, dict):
            continue
        kind = it.get("kind")
        pos = {k: _clamp(it[k], *POS_RANGE_MM, 0.0) for k in ("x_mm", "y_mm") if k in it}
        if kind == "text" and str(it.get("text", "")).strip():
            out.append({"kind": "text", "text": str(it["text"])[:80],
                        "height_mm": _clamp(it.get("height_mm"), *SIZE_RANGE_MM, 30.0), **pos})
        elif kind == "shape" and it.get("shape") in SHAPES:
            item = {"kind": "shape", "shape": it["shape"], "size_mm": _clamp(it.get("size_mm"), *SIZE_RANGE_MM, 50.0),
                    **pos}
            if it.get("n") is not None:
                item["n"] = int(_clamp(it["n"], 3, 24, 5))
            out.append(item)
    if not out:
        raise ValueError("The request didn't produce anything drawable.")
    return out


# ----------------------------------------------------------------------------- planners
def rule_plan(request: str) -> list[dict]:
    """Offline fallback: split on 'and' / 'then' / commas outside quotes, parse each clause."""
    seps = list(re.finditer(r"\s*(?:,|\band then\b|\bthen\b|\band\b)\s*", request))
    # keep only separators that sit outside quotes
    merged, start = [], 0
    for m in seps:
        if len(re.findall(r"[\"“”]|(?<!\w)'|'(?!\w)", request[:m.start()])) % 2 == 0:
            merged.append(request[start:m.start()])
            start = m.end()
    merged.append(request[start:])
    items = []
    for c in merged:
        c = c.strip()
        if not c:
            continue
        if not re.match(r"^\s*(write|text|print|draw)\b", c, re.I):
            c = ("write " + c) if re.search(r"[\"'“‘]", c) else ("draw " + c)
        cmd = parse_command(c)
        if cmd["kind"] == "text":
            items.append({"kind": "text", "text": cmd["text"], "height_mm": cmd["height_mm"]})
        else:
            items.append({"kind": "shape", "shape": cmd["shape"], "size_mm": cmd["size_mm"],
                          **({"n": cmd["n"]} if cmd["n"] else {})})
    return validate_plan(items)


def llm_plan(request: str, client=None, model: str | None = None) -> list[dict]:
    """Claude tool use with a strict schema. `client` can be injected (tests use a fake)."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model or os.environ.get("SYNAPSEARM_LLM_MODEL", DEFAULT_MODEL),
        max_tokens=1024, system=SYSTEM, tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": PLAN_TOOL["name"]},
        messages=[{"role": "user", "content": request}])
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == PLAN_TOOL["name"]:
            return validate_plan(block.input.get("items", []))
    raise ValueError("Model did not return a drawing plan.")


def plan_request(request: str, use_llm: bool | None = None, client=None) -> tuple[list[dict], str]:
    """Returns (plan, planner_used). use_llm=None: use Claude if a key is configured."""
    if use_llm is None:
        use_llm = client is not None or bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_llm:
        try:
            return llm_plan(request, client=client), "llm"
        except Exception as e:  # network, auth, missing package: fall back, but say so
            if client is not None:
                raise
            print(f"[pathgen] LLM planner unavailable ({type(e).__name__}); using rule-based parser")
    return rule_plan(request), "rules"


# ----------------------------------------------------------------------------- layout
def plan_to_strokes(plan, gap_mm=10.0):
    """Render each item, then place it at its x/y (or next in a row, baseline-aligned)."""
    all_strokes, cursor_x = [], 0.0
    for it in plan:
        if it["kind"] == "text":
            strokes = sources.text_strokes(it["text"], it["height_mm"])
        else:
            kw = {}
            if it.get("n"):
                kw = {"points": it["n"]} if it["shape"] == "star" else {"sides": it["n"]}
            strokes = sources.shape_strokes(it["shape"], it["size_mm"], **kw)
        strokes = sources.normalise(strokes)                 # bottom-left at (0, 0), size kept
        w = max(s[:, 0].max() for s in strokes)
        x = it.get("x_mm", cursor_x)
        y = it.get("y_mm", 0.0)
        all_strokes += [s + [x, y] for s in strokes]
        cursor_x = max(cursor_x, x + w + gap_mm)
    return all_strokes
