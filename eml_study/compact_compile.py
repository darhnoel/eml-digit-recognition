"""Compile a tanh network into one pure E-formula, with few E's per weight.

Leaves are only 1 and the pixels; every weight is built from 1. The saving
over the paper's compiler is in "weight times input, then add":

    term          a = E(ln ln x, 1/|w|)   so exp(a) = |w| x      (1 E)
    running sum   V = E(a, exp(V_old))    =  exp(a) - V_old      (2 E's)

The subtraction flips the sign each term enters with, so a term's sign goes
on its input: ln ln(-x) gives exp(a) = -|w| x through the principal complex
logarithm. Not on the weight: negating the large constant 1/|w| would need
exp(1/|w|), which overflows. Chains of `chunk` terms are added with balanced
addition, which keeps the depth moderate.

Run as a script, it builds the compact MNIST formula (paper, Section 6): the
network of `stages` (results/compact/network.npz, rebuilt if missing),
compiled with ordered signs and 4-bit weights, then simplified by merging
equal subexpressions. Writes results/mnist/formula_compact.txt and
results/compact/summary.json, plus NumPy's scores for eml-eval to compare with.

    uv run python -m eml_study.compact_compile
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from eml_study.eml_compiler import Builder
from eml_study.formula import Formula
from eml_study.mnist_data import load
from eml_study.networks import MLP, magnitude_mask, prune_gradually, saliency_mask, train

ROOT = Path(__file__).resolve().parent.parent
NETWORK = ROOT / "results/compact/network.npz"
FORMULA = ROOT / "results/mnist/formula_compact.txt"
BITS = 4  # significant bits per weight in the compact formula


def quantize(w: float, bits: int = 9) -> tuple[int, int]:
    """|w| rounded to n / 2^k with 2^(bits-1) <= n < 2^bits."""
    k = bits - 1 - math.floor(math.log2(abs(w)))
    n = round(abs(w) * 2.0**k)
    if n >= 1 << bits:  # rounding carried into the next power of two
        n, k = n // 2, k - 1
    return n, k


def quantized(W: np.ndarray, bits: int = 9) -> np.ndarray:
    """Weights as the formula stores them."""

    def one(w: float) -> float:
        if w == 0:
            return 0.0
        n, k = quantize(w, bits)
        return math.copysign(n * 2.0**-k, w)

    return np.vectorize(one, otypes=[float])(W)


def compile_compact(net: MLP, chunk: int = 28, bits: int = 9, order_signs: bool = False) -> tuple[Formula, MLP]:
    """The formula, and the network with quantized weights that it computes exactly.

    With `order_signs`, no input is ever negated: each chain alternates positive
    and negative weights, so the running sum's own sign flips do the work, and
    a chain that runs out of one sign negates its sum (see `_ordered_chain`).
    """
    c = _Compiler(chunk, bits, order_signs)
    with c.part("pixels"):
        pixels = [c.lnln(c.var(i)) for i in range(net.W1.shape[1])]
    hidden = []
    for w, b in zip(net.W1, net.b1):
        with c.part("layer 1 sums"):
            s = c.weighted_sum(pixels, w, b)
        with c.part("tanh"):
            h = c.tanh(s)
        with c.part("hidden"):
            hidden.append(c.lnln(h))
    with c.part("layer 2 sums"):
        scores = [c.weighted_sum(hidden, w, b) for w, b in zip(net.W2, net.b2)]
    formula = Formula.from_builder(c, scores, net.W1.shape[1], c.made_by)
    return formula, MLP(*(quantized(p, bits) for p in net))


def compile_ternary(net: MLP, chunk: int = 28, bits: int = 9) -> tuple[Formula, MLP]:
    """For a network whose W1 rows are alpha_j times -1, 0 or +1.

    Every layer-1 term is then +-x/255, so all terms share one constant (255)
    and cost 2 E's each; a hidden unit multiplies its sum by 255 alpha_j once.
    (x/255 rather than x keeps the running sums far below exp's overflow.)
    """
    alpha = np.abs(net.W1).max(axis=1)
    T = np.rint(net.W1 / np.where(alpha > 0, alpha, 1)[:, None])
    assert np.isin(T, (-1, 0, 1)).all(), "layer 1 must be ternary per row"
    c = _Compiler(chunk, bits)
    over_255 = c.integer(255)
    pixels = [c.lnln(c.var(i)) for i in range(net.W1.shape[1])]
    W1q, b1q, hidden = np.zeros_like(net.W1), np.zeros_like(net.b1), []
    for j, (t, b) in enumerate(zip(T, net.b1)):
        S = c.weighted_sum(pixels, t, 0, inverse=over_255)  # sum of t_i x_i / 255
        A, aq = c.constant(255 * alpha[j])
        p = c.mul(A, S)
        W1q[j] = aq / 255 * t
        if b != 0:
            B, b1q[j] = c.constant(b)
            p = c.add(p, B)
        hidden.append(c.lnln(c.tanh(p)))
    scores = [c.weighted_sum(hidden, w, b) for w, b in zip(net.W2, net.b2)]
    quantized_net = MLP(W1q, b1q, quantized(net.W2, bits), quantized(net.b2, bits))
    return Formula.from_builder(c, scores, net.W1.shape[1]), quantized_net


class _Compiler(Builder):
    """A Builder with the compact building blocks."""

    def __init__(self, chunk: int, bits: int, order_signs: bool = False) -> None:
        self.tag, self.made_by = "setup", {}  # which part of the build made each node
        super().__init__()
        self.chunk, self.bits, self.order_signs = chunk, bits, order_signs

    def _node(self, kind: int, a: int, b: int) -> int:
        k = super()._node(kind, a, b)
        self.made_by.setdefault(k, self.tag)
        return k

    @contextmanager
    def part(self, name: str):
        """Label the nodes made inside the block (for Formula.breakdown)."""
        outer, self.tag = self.tag, name
        try:
            yield
        finally:
            self.tag = outer

    def lnln(self, x: int) -> tuple[int, int]:
        """(ln ln x, ln ln(-x)): exp of these is ln x and ln x + i pi."""
        positive = self.ln(self.ln(x))
        if self.order_signs:  # the negated input is never needed
            return positive, None
        with self.part(f"{self.tag}: ln ln(-x)"):
            return positive, self.ln(self.ln(self.neg(x)))

    def inverse(self, w: float) -> int:
        """1/|w| after quantizing, built in log space: 2^k / n."""
        n, k = quantize(w, self.bits)
        with self.part("weight constants"):
            ln_n = self.ln(self.integer(n))
            k_ln2 = self.ln_pow2(k) if k >= 0 else self.neg(self.ln_pow2(-k))
            return self.exp(self.sub(k_ln2, ln_n))

    def constant(self, w: float) -> tuple[int, float]:
        """w itself (sign included), and its quantized value."""
        L, wq = self.ln_abs_quantized(w, self.bits)
        node = self.exp(L)
        return (node if w > 0 else self.neg(node)), math.copysign(wq, w)

    def tanh(self, p: int) -> int:
        """tanh p = 1 - 2 / (e^(2p) + 1)."""
        e2p = self.exp(self.add(p, p))
        return self.sub(self.one, self.exp(self.sub(self.ln(self.integer(2)), self.ln(self.add(e2p, self.one)))))

    def weighted_sum(self, inputs, weights, bias, inverse: int | None = None) -> int:
        """Sum of w x over the nonzero weights, plus the bias.

        `inputs` are lnln pairs. Each term's 1/|w| is built from w, or is the
        shared node `inverse` (then only the sign of w is used).
        """
        terms = [(x, w) for x, w in zip(inputs, weights) if w != 0]
        if bias != 0:
            terms.append((self.lnln(self.one), bias))
        if self.order_signs:
            pos = [t for t in terms if t[1] > 0]
            neg = [t for t in terms if t[1] < 0]
            mixed = [t for pair in zip(pos, neg) for t in pair] + pos[len(neg) :] + neg[len(pos) :]
            chains = [self._ordered_chain(mixed[i : i + self.chunk], inverse) for i in range(0, len(mixed), self.chunk)]
            with self.part(f"{self.tag}: joining chains"):
                return self._join(chains)
        chains = [self._chain(terms[i : i + self.chunk], inverse) for i in range(0, len(terms), self.chunk)]
        with self.part(f"{self.tag}: joining chains"):
            return self.sum(chains)

    def _ordered_chain(self, terms, inverse: int | None) -> tuple[int, int]:
        """Sum of w x without negated inputs; returns (V, sign), V = sign * sum.

        A step V' = E(a, exp V) = |w| x - V. If V holds -S, a positive w gives
        V' = S + w x; if V holds +S, a negative w gives V' = -(S + w x). So the
        signs must alternate; when one sign runs out, V is negated (2 E's).
        """
        groups = {1: [t for t in terms if t[1] > 0], -1: [t for t in terms if t[1] < 0]}
        V, sign = None, 0
        while groups[1] or groups[-1]:
            if V is None:  # start with the larger group: the chain then ends on it
                want = 1 if len(groups[1]) >= len(groups[-1]) else -1
            else:
                want = -sign
                if not groups[want]:
                    with self.part(f"{self.tag}: sign flips"):
                        V, sign = self.neg(V), -sign
                    want = -sign
            (ln_x, _), w = groups[want].pop(0)
            R = self.inverse(w) if inverse is None else inverse
            a = self.eml(ln_x, R)  # exp(a) = |w| x
            V = self.eml(a, self.one if V is None else self.exp(V))
            sign = want
        return V, sign

    def _join(self, chains: list[tuple[int, int]]) -> int:
        """Add chains given as (V, sign): positive ones added, negative ones subtracted."""
        plus = [V for V, s in chains if s > 0]
        minus = [V for V, s in chains if s < 0]
        if not minus:
            return self.sum(plus)
        if not plus:
            return self.neg(self.sum(minus))
        return self.sub(self.sum(plus), self.sum(minus))

    def _chain(self, terms, inverse: int | None) -> int:
        V = None
        for i, ((ln_pos, ln_neg), w) in enumerate(terms):
            s = 1 if (len(terms) - 1 - i) % 2 == 0 else -1  # sign this term enters with
            R = self.inverse(w) if inverse is None else inverse
            a = self.eml(ln_pos if s * w > 0 else ln_neg, R)  # exp(a) = s w x
            V = self.eml(a, self.one if V is None else self.exp(V))
        return V


def probe_images(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """200 training digits and 56 images of random noise, for merge_equal."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (56, X.shape[1])).astype(float)
    return np.concatenate([X[rng.choice(len(X), 200, replace=False)], noise])


def stages(Xtr, ytr, seed: int = 0):
    """The networks on the way to the compact formula (paper, Section 6), as
    (name, network); each step retrains with weights rounded to BITS bits."""
    net = dict(prune_gradually(Xtr, ytr, 64, seed=seed))[0.02]
    yield "pruned layer 1 (2%)", net
    kept = lambda W: (W != 0).astype(float)  # noqa: E731
    net = train(Xtr, ytr, 64, epochs=12, init=net, mask=kept(net.W1), bits=BITS, seed=seed)
    yield f"{BITS}-bit weights", net
    for keep in (0.7, 0.5):
        net = train(Xtr, ytr, 64, epochs=12, init=net, mask=kept(net.W1), mask2=magnitude_mask(net.W2, keep),
                    bits=BITS, seed=seed)
        yield f"layer 2 kept {keep:.0%}", net
    spread = Xtr.std(0) / 255  # the folded W1 reads raw pixels 0..255
    for n in (900, 800, 700):
        net = train(Xtr, ytr, 64, epochs=12, init=net, mask=saliency_mask(net.W1, spread, n),
                    mask2=kept(net.W2), bits=BITS, seed=seed)
        yield f"layer 1 kept {n} weights", net


def build(net: MLP, probes: np.ndarray) -> tuple[Formula, Formula, MLP]:
    """The compact formula of a network: (compiled, merged, quantized network)."""
    compiled, quantized_net = compile_compact(net, bits=BITS, order_signs=True)
    return compiled, compiled.merge_equal(probes), quantized_net


def main():
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    if not NETWORK.exists():
        NETWORK.parent.mkdir(parents=True, exist_ok=True)
        *_, (_, final) = stages(Xtr, ytr)
        final.save(NETWORK)
    net = MLP.load(NETWORK)
    compiled, formula, quantized_net = build(net, probe_images(Xtr))

    before, after = compiled.scores(Xte), formula.scores(Xte)
    assert (after.argmax(1) == before.argmax(1)).all(), "merging changed some answers"
    assert (before.argmax(1) == quantized_net(Xte).argmax(1)).all(), "the formula differs from its network"
    summary = {
        "nonzero_weights": net.n_weights,
        "pixels_read": int(np.count_nonzero(np.abs(net.W1).sum(0))),
        "bits": BITS,
        "E_compiled": len(compiled),
        "E_merged": len(formula),
        "depth": formula.depth,
        "accuracy": float((after.argmax(1) == yte).mean()),
        "max_score_change_from_merging": float(np.abs(after - before).max()),
        "breakdown": formula.breakdown(),
    }
    formula.write(FORMULA)
    after.astype("<f8").tofile(NETWORK.parent / "numpy_scores.f64")
    (NETWORK.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

