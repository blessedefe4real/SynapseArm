import re

import numpy as np
import pytest

from pathgen import optimize, sources
from pathgen.pipeline import build, parse_command
from pathgen.rapid import rapid_stats, to_rapid
from pathgen.reach import Workcell, check_reach


@pytest.mark.parametrize("cmd,expected", [
    ("draw a 60 mm star", {"kind": "shape", "shape": "star", "size_mm": 60.0}),
    ("Draw a hexagon 8 cm", {"kind": "shape", "shape": "hexagon", "size_mm": 80.0}),
    ("write 'Hello robot' 25 mm", {"kind": "text", "text": "Hello robot", "height_mm": 25.0}),
    ("write SynapseArm", {"kind": "text", "text": "SynapseArm", "height_mm": 30.0}),
])
def test_parse_command(cmd, expected):
    out = parse_command(cmd)
    assert {k: out[k] for k in expected} == expected


def test_parse_command_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_command("make me a sandwich")


def test_rdp_respects_tolerance_and_keeps_endpoints():
    t = np.linspace(0, 2 * np.pi, 500)
    curve = np.c_[50 * np.cos(t), 50 * np.sin(t)]
    s = optimize.rdp(curve, 0.1)
    assert len(s) < 0.3 * len(curve)
    assert np.allclose(s[0], curve[0]) and np.allclose(s[-1], curve[-1])
    # every original point within tol of the simplified polyline
    for p in curve[::7]:
        d = min(_seg_dist(p, a, b) for a, b in zip(s[:-1], s[1:]))
        assert d <= 0.1 + 1e-9


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def test_order_strokes_reduces_travel_and_preserves_strokes():
    rng = np.random.default_rng(0)
    strokes = [np.cumsum(rng.normal(0, 2, (5, 2)), 0) + rng.uniform(0, 200, 2) for _ in range(40)]
    ordered = optimize.order_strokes(strokes)
    assert optimize.pen_up_travel(ordered) < 0.5 * optimize.pen_up_travel(strokes)
    assert np.isclose(optimize.draw_length(ordered), optimize.draw_length(strokes))
    # same set of strokes (allowing reversal)
    key = lambda s: tuple(np.round(min(s[0].tolist(), s[-1].tolist()), 6))  # noqa: E731
    assert sorted(map(key, ordered)) == sorted(map(key, strokes))


def test_two_opt_never_worse_than_greedy():
    rng = np.random.default_rng(1)
    strokes = [rng.uniform(0, 100, (2, 2)) for _ in range(30)]
    g = optimize.pen_up_travel(optimize.order_strokes(strokes, two_opt=False))
    assert optimize.pen_up_travel(optimize.order_strokes(strokes)) <= g + 1e-9


def test_shapes_sized_correctly():
    for name in ["circle", "star", "hexagon", "heart"]:
        (s,) = sources.shape_strokes(name, 60)
        assert np.isclose(np.ptp(s[:, 0]), 60)


def test_skeleton_tracing_recovers_a_drawn_line_and_circle():
    cv2 = pytest.importorskip("cv2")
    img = np.full((200, 300), 255, np.uint8)
    cv2.line(img, (20, 100), (120, 100), 0, 5)
    cv2.circle(img, (220, 100), 50, 0, 5)
    strokes = sources.image_strokes(img, width_mm=100)
    lengths = sorted(optimize.draw_length([s]) for s in strokes)
    assert len(strokes) == 2
    # circle circumference / line length = 2*pi*50 / 100
    assert lengths[1] / lengths[0] == pytest.approx(2 * np.pi * 50 / 100, rel=0.15)


def test_rapid_is_well_formed():
    code = to_rapid(sources.shape_strokes("star", 40))
    assert code.count("MODULE") == 2 and code.strip().endswith("ENDMODULE")
    assert code.count("PROC ") == code.count("ENDPROC")
    for ln in code.splitlines():
        s = ln.strip()
        if s and not s.startswith("!") and not re.match(r"^(MODULE|ENDMODULE|PROC|ENDPROC)", s):
            assert s.endswith(";"), s
    st = rapid_stats(code)
    assert st["MoveL"] == 11 + 1 and st["MoveJ"] == 3     # 11 star points + lift; approach + 2 home


def test_reachability_flags_points_outside_workspace():
    stroke = [np.array([[0.0, 0.0], [50.0, 0.0]])]
    assert check_reach(stroke).ok
    far = Workcell(paper_origin_mm=(3000.0, 0.0, 250.0))
    assert check_reach(stroke, far).n_unreachable > 0


def test_end_to_end_text_job():
    job = build("write 'Hi' 20 mm")
    assert job.stats["reachability"]["unreachable"] == 0
    assert job.stats["points_simplified"] <= job.stats["points_raw"]
    assert "MoveL" in job.rapid


# ----------------------------------------------------------------------------- natural-language planner
from types import SimpleNamespace  # noqa: E402

from pathgen.llm import PLAN_TOOL, llm_plan, plan_request, plan_to_strokes, rule_plan, validate_plan  # noqa: E402


class FakeClaude:
    """Stands in for anthropic.Anthropic(): records the call, returns a canned tool_use block."""

    def __init__(self, items):
        self.items, self.calls = items, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="drawing_plan",
                                                        input={"items": self.items})])


def test_llm_plan_uses_forced_tool_and_validates_output():
    fake = FakeClaude([{"kind": "text", "text": "Ada", "height_mm": 30},
                       {"kind": "shape", "shape": "star", "size_mm": 9999, "x_mm": 80, "n": 5},
                       {"kind": "shape", "shape": "dragon"},          # unsupported -> dropped
                       {"kind": "text", "text": "   "}])               # empty -> dropped
    plan = llm_plan("sign Ada with a star", client=fake)
    call = fake.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "drawing_plan"}
    assert call["tools"] == [PLAN_TOOL]
    assert [i["kind"] for i in plan] == ["text", "shape"]
    assert plan[1]["size_mm"] == 400.0                                  # clamped


def test_plan_request_prefers_injected_client_and_falls_back_without_key(monkeypatch):
    plan, who = plan_request("anything", client=FakeClaude([{"kind": "shape", "shape": "circle"}]))
    assert who == "llm" and plan[0]["shape"] == "circle"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    plan, who = plan_request("draw a 40 mm hexagon")
    assert who == "rules" and plan[0]["shape"] == "hexagon"


def test_rule_plan_multiple_items_and_quotes():
    plan = rule_plan("a 60 mm star and the word 'rock and roll' 20 mm")
    assert [i["kind"] for i in plan] == ["shape", "text"]
    assert plan[1]["text"] == "rock and roll"


def test_validate_plan_rejects_empty():
    with pytest.raises(ValueError):
        validate_plan([{"kind": "shape", "shape": "unicorn"}])


def test_layout_places_items_in_a_row_without_overlap():
    strokes = plan_to_strokes([{"kind": "shape", "shape": "circle", "size_mm": 40},
                               {"kind": "shape", "shape": "square", "size_mm": 30}], gap_mm=10)
    xs = [s[:, 0] for s in strokes]
    assert xs[0].max() == pytest.approx(40) and xs[1].min() == pytest.approx(50)
