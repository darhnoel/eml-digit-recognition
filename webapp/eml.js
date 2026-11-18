// Evaluate a pure EML formula file in the browser (or Node).
//
// The file has one line per node:  eN = E(x, y)   with  E(x, y) = exp(x) - ln(y)
// and ends with ten lines  score_k = eN. Leaves are "1" and p1..pD (pixels).
// Arithmetic is complex (principal branch), with the IEEE/C99 special values
// the formula relies on: ln 0 = -Infinity, exp(-Infinity) = 0.

const LINE = /^(e\d+) = E\((\w+), (\w+)\)$/;
const SCORE = /^score_(\d) = (e\d+)$/;

/** Parse formula text into flat arrays: node k computes E(slot a[k], slot b[k]). */
export function parseFormula(text) {
  const lines = text.split("\n");
  let nVars = 0;
  const pending = [];
  const scores = [];
  for (const line of lines) {
    let m = LINE.exec(line);
    if (m) {
      pending.push([m[2], m[3]]);
      for (const v of [m[2], m[3]]) {
        if (v[0] === "p") nVars = Math.max(nVars, parseInt(v.slice(1), 10));
      }
      continue;
    }
    m = SCORE.exec(line);
    if (m) scores[parseInt(m[1], 10)] = m[2];
  }
  // slots: 0 = the constant 1, 1..nVars = pixels, nVars+k = node e_k
  const slot = (name) =>
    name === "1" ? 0
      : name[0] === "p" ? parseInt(name.slice(1), 10)
        : nVars + parseInt(name.slice(1), 10);
  const n = pending.length;
  const a = new Int32Array(n);
  const b = new Int32Array(n);
  // depth level of each slot (leaves 0); nodes in one level are independent
  const level = new Int32Array(1 + nVars + n);
  const levelWidths = [];
  for (let k = 0; k < n; k++) {
    a[k] = slot(pending[k][0]);
    b[k] = slot(pending[k][1]);
    const lv = 1 + Math.max(level[a[k]], level[b[k]]);
    level[1 + nVars + k] = lv;
    levelWidths[lv - 1] = (levelWidths[lv - 1] || 0) + 1;
  }
  return { a, b, nVars, nodes: n, scoreSlots: scores.map(slot), levelWidths };
}

/** Name of a slot as written in the formula file ("1", "p12", "e345"). */
export function slotName(prog, s) {
  if (s === 0) return "1";
  if (s <= prog.nVars) return `p${s}`;
  return `e${s - prog.nVars}`;
}

/** Evaluate the formula for one image. Returns the ten real scores. */
export function evaluate(prog, pixels) {
  return evaluateFull(prog, pixels).scores;
}

/**
 * Evaluate the formula for one image. pixels: array of nVars numbers (0..255).
 * Returns { scores, re, im }: the ten real scores and every slot's value.
 */
export function evaluateFull(prog, pixels) {
  const size = 1 + prog.nVars + prog.nodes;
  const re = new Float64Array(size);
  const im = new Float64Array(size);
  re[0] = 1;
  for (let i = 0; i < prog.nVars; i++) re[1 + i] = pixels[i];
  const { a, b, nodes } = prog;
  const base = 1 + prog.nVars;
  for (let k = 0; k < nodes; k++) {
    const ia = a[k];
    const ib = b[k];
    // exp(x + iy)
    const x = re[ia];
    const y = im[ia];
    let er;
    let ei;
    if (y === 0) {
      er = Math.exp(x);
      ei = 0;
    } else if (x === -Infinity) {
      er = 0;
      ei = 0;
    } else {
      const m = Math.exp(x);
      er = m * Math.cos(y);
      ei = m * Math.sin(y);
    }
    // ln(u + iv), principal branch
    const u = re[ib];
    const v = im[ib];
    let lr;
    let li;
    if (v === 0) {
      lr = Math.log(Math.abs(u));
      li = u < 0 || Object.is(u, -0) ? Math.PI : 0;
    } else {
      lr = Math.log(Math.hypot(u, v));
      li = Math.atan2(v, u);
    }
    re[base + k] = er - lr;
    im[base + k] = ei - li;
  }
  return { scores: prog.scoreSlots.map((s) => re[s]), re, im };
}
