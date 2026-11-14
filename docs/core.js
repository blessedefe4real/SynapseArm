/* SynapseArm demo core: kinematics, IK, path optimisation and RAPID generation in plain JS.
   A port of robotics/kinematics.py and pathgen/ (same D-H model, same algorithms). Runs in the
   browser and in Node (tests: node docs/test_core.js). Units: mm and radians. */
(function (root) {
  "use strict";
  const PI = Math.PI;
  const DH = {
    a: [150, 760, 140, 0, 0, 0],
    alpha: [PI / 2, 0, PI / 2, -PI / 2, PI / 2, 0],
    d: [505, 0, 0, 795, 0, 0],
    qMin: [-180, -100, -165, -270, -130, -400].map(v => v * PI / 180),
    qMax: [180, 155, 255, 270, 130, 400].map(v => v * PI / 180),
  };

  // ---------------------------------------------------------------- small linear algebra
  const mat4mul = (A, B) => {
    const C = new Array(16).fill(0);
    for (let i = 0; i < 4; i++) for (let k = 0; k < 4; k++) {
      const a = A[i * 4 + k]; if (a === 0) continue;
      for (let j = 0; j < 4; j++) C[i * 4 + j] += a * B[k * 4 + j];
    }
    return C;
  };
  const eye4 = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const norm = a => Math.hypot(a[0], a[1], a[2]);
  const col = (T, j) => [T[j], T[4 + j], T[8 + j]];

  function solve(A, b) { // Gaussian elimination with partial pivoting, A n x n (array of rows)
    const n = b.length, M = A.map((r, i) => r.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) {
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
      x[r] = s / M[r][r];
    }
    return x;
  }

  function symEigenvalues(S) { // cyclic Jacobi, symmetric n x n
    const n = S.length, A = S.map(r => r.slice());
    for (let sweep = 0; sweep < 50; sweep++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-20) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-30) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
        }
      }
    }
    return A.map((r, i) => r[i]);
  }

  // ---------------------------------------------------------------- kinematics
  function dhMatrix(th, d, a, al) {
    const ct = Math.cos(th), st = Math.sin(th), ca = Math.cos(al), sa = Math.sin(al);
    return [ct, -st * ca, st * sa, a * ct, st, ct * ca, -ct * sa, a * st, 0, sa, ca, d, 0, 0, 0, 1];
  }
  function frames(q) {
    let T = eye4(); const out = [T];
    for (let i = 0; i < 6; i++) { T = mat4mul(T, dhMatrix(q[i], DH.d[i], DH.a[i], DH.alpha[i])); out.push(T); }
    return out;
  }
  const fk = q => frames(q)[6];

  function jacobian(q, F) {
    F = F || frames(q);
    const on = col(F[6], 3), J = [[], [], [], [], [], []];
    for (let i = 0; i < 6; i++) {
      const z = col(F[i], 2), o = col(F[i], 3), v = cross(z, sub(on, o));
      for (let r = 0; r < 3; r++) { J[r][i] = v[r]; J[r + 3][i] = z[r]; }
    }
    return J;
  }

  function dexterity(q, L = 760) { // inverse condition number of the length-scaled Jacobian
    const J = jacobian(q).map((r, i) => (i < 3 ? r.map(v => v / L) : r));
    const JtJ = [...Array(6)].map((_, i) => [...Array(6)].map((_, j) => J.reduce((s, r) => s + r[i] * r[j], 0)));
    const ev = symEigenvalues(JtJ).map(v => Math.sqrt(Math.max(v, 0))).sort((a, b) => a - b);
    return ev[5] > 0 ? ev[0] / ev[5] : 0;
  }

  function poseError(T, Tt) {
    const ep = [Tt[3] - T[3], Tt[7] - T[7], Tt[11] - T[11]];
    let eo = [0, 0, 0];
    for (let j = 0; j < 3; j++) { const c = cross(col(T, j), col(Tt, j)); eo = [eo[0] + c[0] / 2, eo[1] + c[1] / 2, eo[2] + c[2] / 2]; }
    return ep.concat(eo);
  }

  function ikDLS(Tt, q0, { maxIter = 100, tolPos = 1e-3, tolRot = 1e-5, damping = 1, w = 500 } = {}) {
    let q = q0.slice();
    const W = [1, 1, 1, w, w, w];
    for (let it = 1; it <= maxIter; it++) {
      const F = frames(q), e = poseError(F[6], Tt);
      if (norm(e.slice(0, 3)) < tolPos && norm(e.slice(3)) < tolRot) return { q, ok: true, it };
      const J = jacobian(q, F).map((r, i) => r.map(v => v * W[i])), ew = e.map((v, i) => v * W[i]);
      const JJt = J.map(ri => J.map(rj => ri.reduce((s, v, k) => s + v * rj[k], 0)));
      for (let i = 0; i < 6; i++) JJt[i][i] += damping * damping;
      const y = solve(JJt, ew);
      for (let k = 0; k < 6; k++) {
        let dq = 0; for (let i = 0; i < 6; i++) dq += J[i][k] * y[i];
        q[k] = Math.min(DH.qMax[k], Math.max(DH.qMin[k], q[k] + dq));
      }
    }
    const e = poseError(fk(q), Tt);
    return { q, ok: norm(e.slice(0, 3)) < tolPos && norm(e.slice(3)) < tolRot, it: maxIter };
  }

  function ikWristAnalytic(p, elbow) { // joints 1-3 that put the wrist centre at p
    const [x, y, z] = p, a1 = DH.a[0], a2 = DH.a[1], a3 = DH.a[2], d1 = DH.d[0], d4 = DH.d[3];
    const q1 = Math.atan2(y, x), r = Math.hypot(x, y) - a1, s = z - d1;
    const L3 = Math.hypot(a3, d4), phi = Math.atan2(d4, a3);
    const D = (r * r + s * s - a2 * a2 - L3 * L3) / (2 * a2 * L3);
    if (Math.abs(D) > 1) return null;
    const g = Math.atan2(Math.sqrt(1 - D * D) * (elbow === "up" ? 1 : -1), D);
    const q2 = Math.atan2(s, r) - Math.atan2(L3 * Math.sin(g), a2 + L3 * Math.cos(g));
    return [q1, q2, g + phi];
  }

  function ikSeeded(Tt) { // global solve: analytic wrist-centre seeds x wrist guesses, then DLS
    const p = [Tt[3], Tt[7], Tt[11]];
    let best = null;
    for (const elbow of ["down", "up"]) {
      const q123 = ikWristAnalytic(p, elbow); if (!q123) continue;
      for (const w of [[0, 1, 0], [0, -1, 0], [PI, 1, 0], [PI / 2, 1, 0], [-PI / 2, 1, 0]]) {
        const r = ikDLS(Tt, q123.concat(w), { maxIter: 150 });
        if (r.ok && (!best || Math.abs(r.q[4]) > Math.abs(best.q[4]))) best = r;
      }
      if (best) return best;
    }
    return { q: null, ok: false };
  }

  // ---------------------------------------------------------------- work cell
  function paperRotation(yawDeg) {
    const c = Math.cos(yawDeg * PI / 180), s = Math.sin(yawDeg * PI / 180);
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
  }
  function flangeTarget(cell, x, y, zUp = 0) { // pen tip on paper (x, y) -> flange pose
    const R = paperRotation(cell.yaw), Rd = [[1, 0, 0], [0, -1, 0], [0, 0, -1]];
    const Rt = R.map(r => [0, 1, 2].map(j => r[0] * Rd[0][j] + r[1] * Rd[1][j] + r[2] * Rd[2][j]));
    const tip = [0, 1, 2].map(i => cell.origin[i] + R[i][0] * x + R[i][1] * y + R[i][2] * zUp);
    const pos = [0, 1, 2].map(i => tip[i] - Rt[i][2] * cell.pen);
    return [Rt[0][0], Rt[0][1], Rt[0][2], pos[0], Rt[1][0], Rt[1][1], Rt[1][2], pos[1],
            Rt[2][0], Rt[2][1], Rt[2][2], pos[2], 0, 0, 0, 1];
  }
  function paperToWorld(cell, x, y, z = 0) {
    const R = paperRotation(cell.yaw);
    return [0, 1, 2].map(i => cell.origin[i] + R[i][0] * x + R[i][1] * y + R[i][2] * z);
  }

  /** Solve the whole job with continuation. Returns per-target q (or null) and summary stats. */
  function planJoints(strokes, cell, zUp = 15) {
    const targets = [];
    strokes.forEach((s, k) => {
      targets.push({ x: s[0][0], y: s[0][1], z: zUp, down: false, stroke: k });
      s.forEach(p => targets.push({ x: p[0], y: p[1], z: 0, down: true, stroke: k }));
      const e = s[s.length - 1];
      targets.push({ x: e[0], y: e[1], z: zUp, down: false, stroke: k });
    });
    let qPrev = null, bad = 0, maxStep = 0, minDex = Infinity;
    const Q = [];
    targets.forEach((t, i) => {
      const T = flangeTarget(cell, t.x, t.y, t.z);
      let r = qPrev ? ikDLS(T, qPrev, { maxIter: 60 }) : { ok: false };
      if (!r.ok) r = ikSeeded(T);
      if (r.ok) {
        if (qPrev) maxStep = Math.max(maxStep, ...r.q.map((v, k) => Math.abs(v - qPrev[k])));
        if (i % Math.max(1, Math.floor(targets.length / 150)) === 0) minDex = Math.min(minDex, dexterity(r.q));
        qPrev = r.q; Q.push(r.q);
      } else { bad++; Q.push(null); }
    });
    return { targets, Q, unreachable: bad, maxStepDeg: maxStep * 180 / PI, minDexterity: isFinite(minDex) ? minDex : 0 };
  }

  // ---------------------------------------------------------------- path optimisation
  function rdp(pts, tol) {
    if (pts.length < 3) return pts.slice();
    const keep = new Uint8Array(pts.length); keep[0] = keep[pts.length - 1] = 1;
    const stack = [[0, pts.length - 1]];
    while (stack.length) {
      const [i, j] = stack.pop(); if (j <= i + 1) continue;
      const [ax, ay] = pts[i], [bx, by] = pts[j], dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy);
      let dmax = -1, m = -1;
      for (let k = i + 1; k < j; k++) {
        const d = L < 1e-12 ? Math.hypot(pts[k][0] - ax, pts[k][1] - ay)
          : Math.abs(dx * (pts[k][1] - ay) - dy * (pts[k][0] - ax)) / L;
        if (d > dmax) { dmax = d; m = k; }
      }
      if (dmax > tol) { keep[m] = 1; stack.push([i, m], [m, j]); }
    }
    return pts.filter((_, k) => keep[k]);
  }
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  function penUpTravel(strokes, start = [0, 0]) {
    let pos = start, t = 0;
    for (const s of strokes) { t += dist(pos, s[0]); pos = s[s.length - 1]; }
    return t;
  }
  function drawLength(strokes) {
    let t = 0; for (const s of strokes) for (let i = 1; i < s.length; i++) t += dist(s[i - 1], s[i]);
    return t;
  }
  function orderStrokes(strokes, start = [0, 0], twoOpt = true) {
    const n = strokes.length; if (n <= 1) return strokes.slice();
    const used = new Uint8Array(n), order = []; let pos = start;
    for (let c = 0; c < n; c++) {
      let best = -1, bestD = Infinity, rev = false;
      for (let i = 0; i < n; i++) {
        if (used[i]) continue;
        const s = strokes[i], dh = dist(pos, s[0]), dt = dist(pos, s[s.length - 1]);
        if (dh < bestD) { bestD = dh; best = i; rev = false; }
        if (dt < bestD) { bestD = dt; best = i; rev = true; }
      }
      used[best] = 1; order.push([best, rev]);
      const s = strokes[best]; pos = rev ? s[0] : s[s.length - 1];
    }
    const ends = ([i, r]) => { const s = strokes[i]; return r ? [s[s.length - 1], s[0]] : [s[0], s[s.length - 1]]; };
    if (twoOpt && n > 2) {
      for (let round = 0; round < 30; round++) {
        let improved = false;
        for (let a = 0; a < n - 1; a++) {
          const prevEnd = a === 0 ? start : ends(order[a - 1])[1];
          for (let b = a + 1; b < n; b++) {
            const next = b + 1 < n ? ends(order[b + 1])[0] : null;
            const sa = ends(order[a])[0], eb = ends(order[b])[1];
            const old = dist(sa, prevEnd) + (next ? dist(next, eb) : 0);
            const nu = dist(eb, prevEnd) + (next ? dist(next, sa) : 0);
            if (nu < old - 1e-9) {
              const seg = order.slice(a, b + 1).reverse().map(([i, r]) => [i, !r]);
              order.splice(a, b - a + 1, ...seg); improved = true;
            }
          }
        }
        if (!improved) break;
      }
    }
    return order.map(([i, r]) => (r ? strokes[i].slice().reverse() : strokes[i]));
  }

  // ---------------------------------------------------------------- RAPID
  function toRapid(strokes, cell, { name = "SynapseDraw", zUp = 15, vDraw = 100, vTravel = 500, comment = "" } = {}) {
    const f = v => v.toFixed(2), P = (x, y, z) => `Offs(pOrigin,${f(x)},${f(y)},${f(z)})`;
    const W = " tPen \\WObj:=wobjPaper;", blocks = [];
    strokes.forEach((s, k) => {
      const [x0, y0] = s[0], [x1, y1] = s[s.length - 1];
      const b = [`! stroke ${k + 1}/${strokes.length}: ${s.length} points`,
        `MoveJ ${P(x0, y0, zUp)}, vTravel, z10,${W}`, `MoveL ${P(x0, y0, 0)}, vPlunge, fine,${W}`];
      for (let i = 1; i < s.length - 1; i++) b.push(`MoveL ${P(s[i][0], s[i][1], 0)}, vDraw, z1,${W}`);
      b.push(`MoveL ${P(x1, y1, 0)}, vDraw, fine,${W}`, `MoveL ${P(x1, y1, zUp)}, vPlunge, z5,${W}`);
      blocks.push(b);
    });
    const procs = []; let cur = [];
    for (const b of blocks) { if (cur.length && cur.length + b.length > 250) { procs.push(cur); cur = []; } cur = cur.concat(b); }
    if (cur.length) procs.push(cur);
    const h = cell.yaw * PI / 360, [ox, oy, oz] = cell.origin;
    const nPts = strokes.reduce((s, x) => s + x.length, 0);
    const L = [`MODULE ${name}`, `    ! Generated by the SynapseArm browser demo`,
      ...(comment ? [`    ! ${comment}`] : []),
      `    ! ${strokes.length} strokes, ${nPts} points, ${procs.length} drawing procedures`,
      `    ! Calibrate wobjPaper to the sheet (bottom-left corner, x right, y up) before running.`, ``,
      `    PERS tooldata tPen := [TRUE,[[0,0,${cell.pen.toFixed(1)}],[1,0,0,0]],[0.3,[0,0,40],[1,0,0,0],0,0,0]];`,
      `    TASK PERS wobjdata wobjPaper := [FALSE,TRUE,"",[[${ox.toFixed(1)},${oy.toFixed(1)},${oz.toFixed(1)}],[${Math.cos(h).toFixed(6)},0,0,${Math.sin(h).toFixed(6)}]],[[0,0,0],[1,0,0,0]]];`,
      `    CONST robtarget pOrigin := [[0,0,0],[0,1,0,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];`,
      `    CONST speeddata vDraw := [${vDraw},500,5000,1000];`, `    CONST speeddata vPlunge := [50,500,5000,1000];`,
      `    CONST speeddata vTravel := [${vTravel},500,5000,1000];`, ``,
      `    PROC main()`, `        ConfL \\Off;`, `        ConfJ \\Off;`,
      `        MoveJ Offs(pOrigin,0,0,${f(4 * zUp)}), vTravel, z10, tPen \\WObj:=wobjPaper;`,
      ...procs.map((_, i) => `        Draw${i + 1};`),
      `        MoveJ Offs(pOrigin,0,0,${f(4 * zUp)}), vTravel, fine, tPen \\WObj:=wobjPaper;`, `    ENDPROC`];
    procs.forEach((b, i) => { L.push(``, `    PROC Draw${i + 1}()`, ...b.map(x => `        ${x}`), `    ENDPROC`); });
    L.push(`ENDMODULE`, ``);
    return L.join("\n");
  }

  // ---------------------------------------------------------------- shapes and text
  function shape(name, size) {
    const pts = [];
    if (name === "star") {
      for (let k = 0; k <= 10; k++) { const a = PI / 2 + k * PI / 5, r = k % 2 ? 0.382 : 1; pts.push([r * Math.cos(a), r * Math.sin(a)]); }
    } else if (name === "heart") {
      for (let k = 0; k <= 180; k++) { const t = k / 180 * 2 * PI; pts.push([16 * Math.sin(t) ** 3, 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)]); }
    } else if (name === "spiral") {
      for (let k = 0; k <= 600; k++) { const u = k / 600 * 6 * PI; pts.push([u * Math.cos(u), u * Math.sin(u)]); }
    } else if (name === "hexagon") {
      for (let k = 0; k <= 6; k++) { const a = k * PI / 3; pts.push([Math.cos(a), Math.sin(a)]); }
    } else { // circle
      for (let k = 0; k <= 120; k++) { const a = k / 120 * 2 * PI; pts.push([Math.cos(a), Math.sin(a)]); }
    }
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const minx = Math.min(...xs), miny = Math.min(...ys), s = size / (Math.max(...xs) - minx);
    return [pts.map(([x, y]) => [(x - minx) * s, (y - miny) * s])];
  }

  /** Marching squares on a binary grid -> closed contour polylines (grid units). */
  function contours(mask, W, H) {
    const at = (x, y) => (x >= 0 && y >= 0 && x < W && y < H ? mask[y * W + x] : 0);
    const segs = new Map(), key = p => p[0] + "," + p[1];
    const add = (a, b) => { (segs.get(key(a)) || segs.set(key(a), []).get(key(a))).push(b); };
    for (let y = -1; y < H; y++) for (let x = -1; x < W; x++) {
      const c = (at(x, y) << 3) | (at(x + 1, y) << 2) | (at(x + 1, y + 1) << 1) | at(x, y + 1);
      const T = [x + 0.5, y], R = [x + 1, y + 0.5], B = [x + 0.5, y + 1], L = [x, y + 0.5];
      const m = { 1: [[L, B]], 2: [[B, R]], 3: [[L, R]], 4: [[R, T]], 5: [[L, T], [R, B]], 6: [[B, T]], 7: [[L, T]],
                  8: [[T, L]], 9: [[T, B]], 10: [[T, R], [B, L]], 11: [[T, R]], 12: [[R, L]], 13: [[R, B]], 14: [[B, L]] }[c];
      if (m) for (const [a, b] of m) add(a, b);
    }
    const out = [];
    for (const [k0] of segs) {
      while (segs.get(k0) && segs.get(k0).length) {
        const path = [k0.split(",").map(Number)];
        let k = k0;
        while (segs.get(k) && segs.get(k).length) {
          const nxt = segs.get(k).pop(); path.push(nxt); k = key(nxt);
          if (k === k0) break;
        }
        if (path.length > 3) out.push(path);
      }
    }
    return out;
  }

  root.SynapseCore = { DH, frames, fk, jacobian, dexterity, poseError, ikDLS, ikSeeded, ikWristAnalytic,
    flangeTarget, paperToWorld, planJoints, rdp, orderStrokes, penUpTravel, drawLength, toRapid, shape, contours, symEigenvalues };
})(typeof module !== "undefined" ? module.exports : window);
