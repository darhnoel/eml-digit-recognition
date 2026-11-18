// Turn a drawing or photo into the 784 values (0..255) the MNIST formula reads,
// following how MNIST itself was prepared: find the ink, crop it, scale the
// longer side to 20 pixels with anti-aliasing (Lanczos), and place it in a 28x28
// frame so that its centre of mass is in the middle.
//
// This reproduces eml_study/image_input.py step by step (NumPy percentiles,
// Python's round-half-to-even, Pillow's two-pass 8-bit Lanczos resize), so the
// browser demo prepares images the same way as the paper's experiments.

/** Python's round(): halves go to the even neighbour. */
function pyRound(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** numpy.percentile with linear interpolation (q in 0..100). */
function percentile(values, q) {
  const s = Float64Array.from(values).sort();
  const pos = (q / 100) * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/** Photo: grayscale (0..255) -> ink strength 0..1 with paper/ink levels detected. */
function inkFromGray(gray, w, h) {
  let border = 0;
  let nb = 0;
  let mean = 0;
  for (let i = 0; i < gray.length; i++) mean += gray[i];
  mean /= gray.length;
  for (let x = 0; x < w; x++) { border += gray[x] + gray[(h - 1) * w + x]; nb += 2; }
  for (let y = 0; y < h; y++) { border += gray[y * w] + gray[y * w + w - 1]; nb += 2; }
  const g = border / nb < mean ? gray.map((v) => 255 - v) : gray; // light ink on dark: flip
  const paper = percentile(g, 90);
  const inkLevel = percentile(g, 1);
  if (paper - inkLevel < 20) return null;
  const ink = new Float64Array(g.length);
  for (let i = 0; i < g.length; i++) {
    const v = Math.min(1, Math.max(0, (paper - g[i]) / (paper - inkLevel)));
    ink[i] = v < 0.25 ? 0 : v;
  }
  return ink;
}

// --- Pillow-compatible Lanczos resize for 8-bit images ------------------------
function lanczos(x) {
  if (x === 0) return 1;
  if (x <= -3 || x >= 3) return 0;
  const px = Math.PI * x;
  return (3 * Math.sin(px) * Math.sin(px / 3)) / (px * px);
}

function coeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 3 * filterscale;
  const out = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    const k = [];
    let total = 0;
    for (let x = 0; x < xmax; x++) {
      const wgt = lanczos((x + xmin - center + 0.5) * ss);
      k.push(wgt);
      total += wgt;
    }
    out.push({ xmin, k: k.map((v) => (total ? v / total : 0)) });
  }
  return out;
}

const clip8 = (v) => Math.min(255, Math.max(0, Math.round(v)));

/** Resize an 8-bit (w x h) image to (W x H): horizontal pass, then vertical. */
function resizeLanczos(src, w, h, W, H) {
  let cur = src;
  let cw = w;
  if (W !== w) {
    const cx = coeffs(w, W);
    const tmp = new Uint8Array(W * h);
    for (let y = 0; y < h; y++) {
      for (let X = 0; X < W; X++) {
        const { xmin, k } = cx[X];
        let s = 0;
        for (let i = 0; i < k.length; i++) s += cur[y * cw + xmin + i] * k[i];
        tmp[y * W + X] = clip8(s);
      }
    }
    cur = tmp;
    cw = W;
  }
  if (H !== h) {
    const cy = coeffs(h, H);
    const tmp = new Uint8Array(cw * H);
    for (let Y = 0; Y < H; Y++) {
      const { xmin, k } = cy[Y];
      for (let x = 0; x < cw; x++) {
        let s = 0;
        for (let i = 0; i < k.length; i++) s += cur[(xmin + i) * cw + x] * k[i];
        tmp[Y * cw + x] = clip8(s);
      }
    }
    cur = tmp;
  }
  return cur;
}

/**
 * ImageData (any size) -> Float64Array(784) with values 0..255, or null if empty.
 * drawn: true for the drawing pad (black ink on white, read directly);
 * false for photos (paper and ink levels detected automatically).
 */
export function toMnist(imageData, { drawn = false } = {}) {
  const { width: w, height: h, data } = imageData;
  const gray = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const a = data[4 * i + 3] / 255;
    const l = Math.round((data[4 * i] * 299 + data[4 * i + 1] * 587 + data[4 * i + 2] * 114) / 1000);
    gray[i] = l * a + 255 * (1 - a); // transparent counts as paper
  }
  // drawn: the pad is transparent and strokes are opaque, so ink = alpha
  // (works whatever colour the strokes are drawn in, e.g. in dark mode)
  const ink = drawn
    ? Float64Array.from({ length: w * h }, (_, i) => data[4 * i + 3] / 255)
    : inkFromGray(gray, w, h);
  if (!ink) return null;

  let r0 = h;
  let r1 = -1;
  let c0 = w;
  let c1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (ink[y * w + x] > 0.5) {
        r0 = Math.min(r0, y); r1 = Math.max(r1, y);
        c0 = Math.min(c0, x); c1 = Math.max(c1, x);
      }
    }
  }
  if (r1 < 0) return null;
  const ch = r1 - r0 + 1;
  const cw = c1 - c0 + 1;
  const crop = new Uint8Array(ch * cw);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) crop[y * cw + x] = Math.floor(ink[(r0 + y) * w + (c0 + x)] * 255);
  }
  const scale = 20 / Math.max(ch, cw);
  const nh = Math.max(1, pyRound(ch * scale));
  const nw = Math.max(1, pyRound(cw * scale));
  const small = resizeLanczos(crop, cw, ch, nw, nh);

  let total = 0;
  let cy = 0;
  let cx = 0;
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const v = small[y * nw + x];
      total += v; cy += y * v; cx += x * v;
    }
  }
  cy /= total;
  cx /= total;
  const top = Math.min(Math.max(pyRound(14 - cy), 0), 28 - nh);
  const left = Math.min(Math.max(pyRound(14 - cx), 0), 28 - nw);
  const frame = new Float64Array(784);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) frame[(top + y) * 28 + (left + x)] = small[y * nw + x];
  }
  return frame;
}
