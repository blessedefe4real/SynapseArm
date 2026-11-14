// Checks the JS port against reference values from the Python implementation.
// Run: node docs/test_core.js   (also run by pytest via tests/test_demo_core.py)
const C = require("./core.js").SynapseCore;
const assert = require("assert");
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);

// FK / Jacobian / dexterity vs robotics/kinematics.py at q = [0.3, -0.4, 0.5, 0.2, 0.9, -0.7]
const q = [0.3, -0.4, 0.5, 0.2, 0.9, -0.7];
const T = C.fk(q);
[[3, 1020.943749], [7, 315.814911], [11, -568.009573], [0, 0.348768], [6, 0.095408], [10, -0.541861]]
  .forEach(([i, v]) => close(T[i], v, 1e-5, `fk[${i}]`));
const J = C.jacobian(q);
[1025.085198, 317.096011, 918.674505, 0.29552, -0.955336, 0].forEach((v, r) => close(J[r][1], v, 1e-5, `J[${r}][1]`));
close(C.dexterity(q), 0.1387183395996362, 1e-9, "dexterity");

// IK round trip from a cold start and from continuation
const Tt = C.fk([0.2, 0.3, 0.1, 0.4, 0.8, -0.3]);
const r = C.ikSeeded(Tt);
assert.ok(r.ok, "ikSeeded converges");
const e = C.poseError(C.fk(r.q), Tt);
assert.ok(Math.hypot(e[0], e[1], e[2]) < 1e-3, "position error < 1 um");

// Work cell: pen points down, tip lands on the rotated paper
const cell = { origin: [800, 0, 300], yaw: 90, pen: 120 };
const F = C.flangeTarget(cell, 100, 0, 0);
close(F[2], 0, 1e-12, "z-axis x"); close(F[6], 0, 1e-12, "z-axis y"); close(F[10], -1, 1e-12, "z-axis z");
close(F[3], 800, 1e-9, "tip x"); close(F[7], 100, 1e-9, "tip y"); close(F[11], 300 + 120, 1e-9, "flange z");

// Path tools
const circle = [...Array(500)].map((_, k) => [50 * Math.cos(k / 499 * 2 * Math.PI), 50 * Math.sin(k / 499 * 2 * Math.PI)]);
assert.ok(C.rdp(circle, 0.1).length < 150, "rdp simplifies");
let seed = 1; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const strokes = [...Array(40)].map(() => { const x = rnd() * 200, y = rnd() * 200; return [[x, y], [x + rnd() * 5, y + rnd() * 5]]; });
const ordered = C.orderStrokes(strokes);
assert.ok(C.penUpTravel(ordered) < 0.5 * C.penUpTravel(strokes), "ordering halves pen-up travel");
close(C.drawLength(ordered), C.drawLength(strokes), 1e-9, "draw length preserved");

// Whole job: a small star is reachable with smooth joint motion
const job = C.planJoints(C.shape("star", 60), { origin: [700, -150, 250], yaw: 0, pen: 120 });
assert.strictEqual(job.unreachable, 0, "star reachable");
assert.ok(job.maxStepDeg < 10, "continuous joint path");

// RAPID structure
const code = C.toRapid(C.shape("star", 40), { origin: [700, -150, 250], yaw: 90, pen: 120 });
assert.ok(code.startsWith("MODULE SynapseDraw") && code.trim().endsWith("ENDMODULE"));
assert.ok(code.includes("[0.707107,0,0,0.707107]"), "wobj quaternion");
assert.strictEqual((code.match(/PROC /g) || []).length, (code.match(/ENDPROC/g) || []).length);

// Marching squares: a filled square gives one closed contour
const W = 20, H = 20, mask = new Uint8Array(W * H);
for (let y = 5; y < 15; y++) for (let x = 5; x < 15; x++) mask[y * W + x] = 1;
const cs = C.contours(mask, W, H);
assert.strictEqual(cs.length, 1, "one contour");
console.log("demo core: all checks passed");
