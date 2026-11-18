"""Compile arithmetic into pure EML graphs and evaluate them.

Grammar (arXiv:2603.21852): S -> 1 | x_i | eml(S, S), eml(a, b) = exp(a) - ln(b).
The builder hash-conses identical subexpressions, so the graph is a DAG.
Its *expanded tree* (every shared subexpression copied out) is the literal
EML expression; its size and depth are reported separately.

Evaluation uses NumPy complex128 with C99/IEEE special values
(ln 0 = -inf, exp(-inf) = 0). PyTorch's complex exp returns NaN for
exp(inf + 0j), which breaks ln(0) through the eml chain, so it is not used here.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

ONE, VAR, EML = 0, 1, 2


class Builder:
    def __init__(self) -> None:
        self.kind: list[int] = []
        self.a: list[int] = []
        self.b: list[int] = []
        self._cache: dict[tuple, int] = {}
        self.one = self._node(ONE, -1, -1)
        self.zero = self.ln(self.one)  # ln 1 = 0
        self._ints: dict[int, int] = {1: self.one}

    # -- core -----------------------------------------------------------------
    def _node(self, kind: int, a: int, b: int) -> int:
        key = (kind, a, b)
        if key in self._cache:
            return self._cache[key]
        self.kind.append(kind)
        self.a.append(a)
        self.b.append(b)
        self._cache[key] = len(self.kind) - 1
        return self._cache[key]

    def var(self, i: int) -> int:
        return self._node(VAR, i, -1)

    def eml(self, x: int, y: int) -> int:
        return self._node(EML, x, y)

    # -- derived operations (identities from the paper) -------------------------
    def exp(self, x):  # e^x = eml(x, 1)
        return self.eml(x, self.one)

    def ln(self, x):  # ln x = eml(1, eml(eml(1, x), 1))
        return self.eml(self.one, self.eml(self.eml(self.one, x), self.one))

    def sub(self, x, y):  # x - y = eml(ln x, exp y)
        return self.eml(self.ln(x), self.exp(y))

    def neg(self, y):
        return self.sub(self.zero, y)

    def add(self, x, y):
        return self.sub(x, self.neg(y))

    def mul(self, x, y):  # x*y = exp(ln x + ln y)
        return self.exp(self.add(self.ln(x), self.ln(y)))

    def div(self, x, y):  # x/y = exp(ln x - ln y)
        return self.exp(self.sub(self.ln(x), self.ln(y)))

    def sum(self, terms: list[int]) -> int:
        """Balanced addition tree (keeps depth logarithmic in the term count)."""
        if not terms:
            return self.zero
        while len(terms) > 1:
            nxt = [self.add(terms[i], terms[i + 1]) for i in range(0, len(terms) - 1, 2)]
            if len(terms) % 2:
                nxt.append(terms[-1])
            terms = nxt
        return terms[0]

    # -- constants built from 1 ------------------------------------------------
    MAX_INT = 700  # x - y = eml(ln x, exp y) needs exp(y) finite: |y| < ~709

    def integer(self, n: int) -> int:
        """Positive integer from 1 by binary doubling and addition (n <= 700).

        Larger values break the subtraction identity in float64, because the
        intermediate exp(-y) underflows to 0 and ln 0 = -inf.
        """
        if n < 1 or n > self.MAX_INT:
            raise ValueError(f"integers 1..{self.MAX_INT} only")
        if n in self._ints:
            return self._ints[n]
        powers, p, node = [], 1, self.one
        while p * 2 <= n:
            node = self._ints.get(p * 2) or self.add(node, node)
            p *= 2
            self._ints[p] = node
        bits = [self._ints[1 << i] for i in range(n.bit_length()) if n >> i & 1]
        out = self.sum(bits)
        self._ints[n] = out
        return out

    def ln_pow2(self, k: int) -> int:
        """k * ln 2 for k >= 0, as a sum of doubled ln 2 terms (values stay small)."""
        if k == 0:
            return self.zero
        key = ("lnpow2", k)
        if key in self._cache:
            return self._cache[key]
        parts, cur, i = [], self.ln(self.integer(2)), 0
        while (1 << i) <= k:
            if k >> i & 1:
                parts.append(cur)
            cur = self.add(cur, cur)
            i += 1
        out = self.sum(parts)
        self._cache[key] = out
        return out

    def ln_abs_quantized(self, w: float, bits: int = 9) -> tuple[int, float]:
        """ln|w| for w != 0, with |w| rounded to n * 2^-k, 2^(bits-1) <= n < 2^bits.

        Returns (node, quantized |w|). Built only from 1, in log space, so every
        intermediate value stays far below the exp overflow limit.
        """
        a = abs(w)
        k = bits - 1 - int(np.floor(np.log2(a)))
        n = int(round(a * 2.0**k))
        if n >= 1 << bits:  # rounding carried into the next power of two
            n //= 2
            k -= 1
        ln_n = self.ln(self.integer(n))
        node = self.sub(ln_n, self.ln_pow2(k)) if k >= 0 else self.add(ln_n, self.ln_pow2(-k))
        return node, n * 2.0**-k

    def ln_fraction(self, f: Fraction) -> int:
        """ln of a positive rational p/q, as an EML node."""
        if f <= 0:
            raise ValueError("positive rationals only")
        lp = self.ln(self.integer(f.numerator))
        if f.denominator == 1:
            return lp
        return self.sub(lp, self.ln(self.integer(f.denominator)))

    # -- measurement -----------------------------------------------------------
    def stats(self, roots: list[int]) -> dict:
        """DAG nodes reachable from roots, expanded-tree size, and depth."""
        reach = set()
        stack = list(roots)
        while stack:
            n = stack.pop()
            if n in reach:
                continue
            reach.add(n)
            if self.kind[n] == EML:
                stack += [self.a[n], self.b[n]]
        size: dict[int, int] = {}
        depth: dict[int, int] = {}
        for n in sorted(reach):  # children always have smaller ids
            if self.kind[n] == EML:
                size[n] = 1 + size[self.a[n]] + size[self.b[n]]
                depth[n] = 1 + max(depth[self.a[n]], depth[self.b[n]])
            else:
                size[n], depth[n] = 1, 0
        eml_nodes = sum(1 for n in reach if self.kind[n] == EML)
        return {
            "dag_nodes": len(reach),
            "dag_eml_nodes": eml_nodes,
            "tree_size_total": sum(size[r] for r in roots),
            "max_depth": max(depth[r] for r in roots),
        }

    # -- evaluation ------------------------------------------------------------
    def _reachable(self, roots: list[int]) -> list[int]:
        need, stack = set(), list(roots)
        while stack:
            k = stack.pop()
            if k in need:
                continue
            need.add(k)
            if self.kind[k] == EML:
                stack += [self.a[k], self.b[k]]
        return sorted(need)

    def evaluate_lowmem(self, roots: list[int], X: np.ndarray) -> np.ndarray:
        """Like evaluate(), but frees each value once its last user is computed.

        Needed for large formulas, where keeping every intermediate value for
        every sample would not fit in memory.
        """
        order = self._reachable(roots)
        uses: dict[int, int] = {}
        for k in order:
            if self.kind[k] == EML:
                uses[self.a[k]] = uses.get(self.a[k], 0) + 1
                uses[self.b[k]] = uses.get(self.b[k], 0) + 1
        keep = set(roots)
        n = X.shape[0]
        Xc = X.astype(np.complex128)
        ones = np.ones(n, dtype=np.complex128)
        val: dict[int, np.ndarray] = {}
        with np.errstate(all="ignore"):
            for k in order:
                kind = self.kind[k]
                if kind == ONE:
                    val[k] = ones
                elif kind == VAR:
                    val[k] = Xc[:, self.a[k]]
                else:
                    a, b = self.a[k], self.b[k]
                    val[k] = np.exp(val[a]) - np.log(val[b])
                    for c in (a, b):
                        uses[c] -= 1
                        if uses[c] == 0 and c not in keep:
                            del val[c]
        return np.stack([val[r] for r in roots], axis=1)

    def write_formula(self, roots: list[int], path, n_vars: int, header: str = "") -> int:
        """Write the formula as one line per E node; returns the number of E lines."""
        order = self._reachable(roots)
        name: dict[int, str] = {}
        lines = 0
        with open(path, "w") as f:
            f.write("# Pure E-formula. E(x, y) = exp(x) - ln(y).\n")
            f.write(f"# Leaves: 1 and p1..p{n_vars} (pixel values, row by row).\n")
            f.write("# Each line names one E(...) so repeated parts are written once.\n")
            f.write("# To read a digit: compute score_0 .. score_9; the answer is the largest.\n")
            if header:
                f.write(f"# {header}\n")
            f.write("\n")
            for k in order:
                if self.kind[k] == ONE:
                    name[k] = "1"
                elif self.kind[k] == VAR:
                    name[k] = f"p{self.a[k] + 1}"
                else:
                    lines += 1
                    name[k] = f"e{lines}"
                    f.write(f"{name[k]} = E({name[self.a[k]]}, {name[self.b[k]]})\n")
            f.write("\n")
            for i, r in enumerate(roots):
                f.write(f"score_{i} = {name[r]}\n")
        return lines

    def evaluate(self, roots: list[int], X: np.ndarray) -> np.ndarray:
        """Evaluate roots for every row of X (n_samples, n_vars). Returns complex (n, len(roots))."""
        n = X.shape[0]
        need = set()
        stack = list(roots)
        while stack:
            k = stack.pop()
            if k in need:
                continue
            need.add(k)
            if self.kind[k] == EML:
                stack += [self.a[k], self.b[k]]
        val: dict[int, np.ndarray] = {}
        Xc = X.astype(np.complex128)
        with np.errstate(all="ignore"):
            for k in sorted(need):
                kind = self.kind[k]
                if kind == ONE:
                    val[k] = np.ones(n, dtype=np.complex128)
                elif kind == VAR:
                    val[k] = Xc[:, self.a[k]]
                else:
                    val[k] = np.exp(val[self.a[k]]) - np.log(val[self.b[k]])
        return np.stack([val[r] for r in roots], axis=1)
