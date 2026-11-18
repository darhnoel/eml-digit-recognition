// Evaluate a pure EML formula file in the browser (or Node).
//
// The file has one line per node:  eN = E(x, y)   with  E(x, y) = exp(x) - ln(y)
// and ends with ten lines  score_k = eN. Leaves are "1" and p1..pD (pixels).
// Arithmetic is complex (principal branch), with the IEEE/C99 special values
// the formula relies on: ln 0 = -Infinity, exp(-Infinity) = 0, and signed
// zeros: ln(-0 - 0i) = -Infinity - i*pi, not + i*pi.
//
// Speed: everything that does not depend on the pixels is computed once, when
// the formula is parsed. E's built only from 1 (the weights) become constants,
// and so does the exp or ln half of an E whose operand is such a constant.
// Per image, only the remaining E's are evaluated.

const LINE = /^(e\d+) = E\((\w+), (\w+)\)$/;
const SCORE = /^score_(\d) = (e\d+)$/;

// results of cexp / cln (module-level so the hot loop allocates nothing)
let RE = 0;
let IM = 0;

/** exp(x + iy) into RE, IM. */
function cexp(x, y) {
  if (y === 0) {
    RE = Math.exp(x);
    IM = y; // keeps the sign of zero
  } else if (!Number.isFinite(y) && Math.abs(x) === Infinity) {
    // C99: exp(-inf + i inf) = 0, exp(+inf + i inf) = inf + i NaN
    RE = x < 0 ? 0 : x;
    IM = x < 0 ? 0 : NaN;
  } else {
    const m = Math.exp(x);
    RE = m * Math.cos(y);
    IM = m * Math.sin(y);
  }
}

/** ln(u + iv), principal branch, into RE, IM. */
function cln(u, v) {
  if (v === 0 && !Number.isNaN(u)) {
    RE = Math.log(Math.abs(u));
    IM = u > 0 || Object.is(u, 0) ? v : Object.is(v, -0) ? -Math.PI : Math.PI;
  } else {
    RE = Math.log(Math.hypot(u, v));
    IM = Math.atan2(v, u);
  }
}

/** Parse formula text and prepare it for fast evaluation. */
export function parseFormula(text) {
  let nVars = 0;
  const pending = [];
  const scores = [];
  for (const line of text.split("\n")) {
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
  const base = 1 + nVars;
  const size = base + n;
  const a = new Int32Array(n);
  const b = new Int32Array(n);
  const level = new Int32Array(size); // depth of each slot (leaves 0)
  const levelWidths = [];
  for (let k = 0; k < n; k++) {
    a[k] = slot(pending[k][0]);
    b[k] = slot(pending[k][1]);
    const lv = 1 + Math.max(level[a[k]], level[b[k]]);
    level[base + k] = lv;
    levelWidths[lv - 1] = (levelWidths[lv - 1] || 0) + 1;
  }

  // Values that are the same for every image: slot 0, and every E built only from 1.
  const isConst = new Uint8Array(size);
  const constRe = new Float64Array(size);
  const constIm = new Float64Array(size);
  isConst[0] = 1;
  constRe[0] = 1;
  // The E's that depend on the pixels, with their constant halves precomputed:
  // value = exp-half - ln-half, a half being a slot to compute or a known number.
  const op = { k: [], a: [], b: [], er: [], ei: [], lr: [], li: [] };
  for (let k = 0; k < n; k++) {
    const ca = isConst[a[k]];
    const cb = isConst[b[k]];
    let er = 0, ei = 0, lr = 0, li = 0;
    if (ca) { cexp(constRe[a[k]], constIm[a[k]]); er = RE; ei = IM; }
    if (cb) { cln(constRe[b[k]], constIm[b[k]]); lr = RE; li = IM; }
    if (ca && cb) {
      isConst[base + k] = 1;
      constRe[base + k] = er - lr;
      constIm[base + k] = ei - li;
      continue;
    }
    op.k.push(base + k);
    op.a.push(ca ? -1 : a[k]); // -1: the exp half is known
    op.b.push(cb ? -1 : b[k]); // -1: the ln half is known
    op.er.push(er); op.ei.push(ei); op.lr.push(lr); op.li.push(li);
  }
  const ops = {
    k: Int32Array.from(op.k), a: Int32Array.from(op.a), b: Int32Array.from(op.b),
    er: Float64Array.from(op.er), ei: Float64Array.from(op.ei),
    lr: Float64Array.from(op.lr), li: Float64Array.from(op.li),
  };
  // One working buffer per formula: constants are written once and never
  // overwritten; each evaluation rewrites only the pixels and the other E's.
  return {
    a, b, nVars, nodes: n, scoreSlots: scores.map(slot), levelWidths,
    perImage: ops.k.length, re: constRe, im: constIm, ops,
  };
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
 * re and im are the formula's working buffer: the next call overwrites them.
 */
export function evaluateFull(prog, pixels) {
  const { re, im } = prog;
  for (let i = 0; i < prog.nVars; i++) { re[1 + i] = pixels[i]; im[1 + i] = 0; }
  const { k, a, b, er, ei, lr, li } = prog.ops;
  for (let j = 0; j < k.length; j++) {
    let xr = er[j], xi = ei[j], yr = lr[j], yi = li[j];
    if (a[j] >= 0) { cexp(re[a[j]], im[a[j]]); xr = RE; xi = IM; }
    if (b[j] >= 0) { cln(re[b[j]], im[b[j]]); yr = RE; yi = IM; }
    re[k[j]] = xr - yr;
    im[k[j]] = xi - yi;
  }
  return { scores: prog.scoreSlots.map((s) => re[s]), re, im };
}
