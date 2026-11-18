"""An E-formula as a straight-line program (the same shape eml-eval reads).

Slot 0 holds the constant 1, slots 1..n_inputs hold the inputs p1, p2, ...,
and E number j (1-based, as in the text file) is stored in slot n_inputs + j,
computing E(slot a, slot b) = exp(a) - ln(b). Operands are always earlier slots.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from eml_study.eml_compiler import EML, ONE, Builder

HEADER = """\
# Pure E-formula. E(x, y) = exp(x) - ln(y).
# Leaves: 1 and p1..p{n} (pixel values, row by row).
# Each line names one E(...) so repeated parts are written once.
# To read a digit: compute score_0 .. score_9; the answer is the largest.

"""
E_LINE = re.compile(r"(e\d+) = E\((\w+), (\w+)\)")
SCORE_LINE = re.compile(r"score_\d+ = (\w+)")


@dataclass(frozen=True, eq=False)
class Formula:
    a: np.ndarray  # first operand slot of each E
    b: np.ndarray  # second operand slot of each E
    outputs: np.ndarray  # slots of score_0, score_1, ...
    n_inputs: int = 784

    # -- building --------------------------------------------------------------
    @classmethod
    def from_builder(cls, B: Builder, roots: list[int], n_inputs: int = 784) -> Formula:
        """The E's of a Builder graph that the roots use, in creation order."""
        used, stack = set(), list(roots)
        while stack:
            k = stack.pop()
            if k not in used:
                used.add(k)
                if B.kind[k] == EML:
                    stack += [B.a[k], B.b[k]]
        es = sorted(k for k in used if B.kind[k] == EML)  # operands are created first
        slot = {k: 0 if B.kind[k] == ONE else B.a[k] + 1 for k in used if B.kind[k] != EML}
        slot.update({k: n_inputs + 1 + j for j, k in enumerate(es)})
        return cls(
            np.array([slot[B.a[k]] for k in es], dtype=np.int64),
            np.array([slot[B.b[k]] for k in es], dtype=np.int64),
            np.array([slot[r] for r in roots], dtype=np.int64),
            n_inputs,
        )

    @classmethod
    def read(cls, path: Path, n_inputs: int = 784) -> Formula:
        slot = {"1": 0} | {f"p{i}": i for i in range(1, n_inputs + 1)}
        a, b, outputs = [], [], []
        for line in Path(path).read_text().splitlines():
            if m := E_LINE.fullmatch(line):
                name, x, y = m.groups()
                a.append(slot[x])
                b.append(slot[y])
                slot[name] = n_inputs + len(a)
            elif m := SCORE_LINE.fullmatch(line):
                outputs.append(slot[m.group(1)])
        return cls(
            np.array(a, dtype=np.int64), np.array(b, dtype=np.int64), np.array(outputs, dtype=np.int64), n_inputs
        )

    def write(self, path: Path) -> None:
        names = ["1"] + [f"p{i}" for i in range(1, self.n_inputs + 1)] + [f"e{j}" for j in range(1, len(self) + 1)]
        es = [f"e{j} = E({names[x]}, {names[y]})\n" for j, (x, y) in enumerate(zip(self.a, self.b), 1)]
        scores = [f"score_{k} = {names[s]}\n" for k, s in enumerate(self.outputs)]
        Path(path).write_text(HEADER.format(n=self.n_inputs) + "".join(es) + "\n" + "".join(scores))

    # -- shape -----------------------------------------------------------------
    def __len__(self) -> int:
        """Number of E's."""
        return len(self.a)

    @property
    def n_slots(self) -> int:
        return self.n_inputs + 1 + len(self)

    @cached_property
    def levels(self) -> np.ndarray:
        """0 for a leaf; for an E, 1 + the level of its deeper operand."""
        level = [0] * (self.n_inputs + 1)
        for x, y in zip(self.a.tolist(), self.b.tolist()):
            level.append(1 + max(level[x], level[y]))
        return np.array(level)

    @property
    def depth(self) -> int:
        return int(self.levels.max())

    def tree_sizes(self) -> list[int]:
        """Size of each slot written out as a plain tree (shared parts copied)."""
        size = [1] * (self.n_inputs + 1)
        for x, y in zip(self.a.tolist(), self.b.tolist()):
            size.append(1 + size[x] + size[y])
        return size

    # -- evaluation ------------------------------------------------------------
    @cached_property
    def _waves(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """E's grouped by level: each group is one vectorized step."""
        first = self.n_inputs + 1
        level = self.levels[first:]
        order = np.argsort(level, kind="stable")
        groups = np.split(order, np.flatnonzero(np.diff(level[order])) + 1)
        return [(first + js, self.a[js], self.b[js]) for js in groups if len(js)]

    def values(self, X: np.ndarray) -> np.ndarray:
        """Every slot's value for every row of X: complex array (n_slots, len(X))."""
        V = np.empty((self.n_slots, len(X)), dtype=np.complex128)
        V[0] = 1
        V[1 : self.n_inputs + 1] = X.T
        with np.errstate(all="ignore"):  # ln 0 = -inf and exp(-inf) = 0 are intended
            for out, a, b in self._waves:
                V[out] = np.exp(V[a]) - np.log(V[b])
        return V

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Real parts of the outputs (n, n_outputs), in batches of about 128 MB."""
        batch = max(1, 2**23 // self.n_slots)
        return np.concatenate([self.values(X[i : i + batch])[self.outputs].real.T for i in range(0, len(X), batch)])

    # -- exact simplification --------------------------------------------------
    def merge_equal(self, probes: np.ndarray) -> Formula:
        """Replace every E that has the same values on `probes` as a smaller,
        earlier node by that node; E's nobody uses any more disappear.

        "Smaller" is the size written out as a tree. Candidates are grouped by
        their first 24 values and confirmed on all probes.
        """
        V = self.values(probes)
        size = self.tree_sizes()
        groups = defaultdict(list)
        for s, v in enumerate(V):
            groups[_fingerprint(v)].append(s)
        rep = np.arange(self.n_slots)
        for slots in groups.values():
            best = min(slots, key=lambda s: (size[s], s))
            for s in slots:
                if s > max(best, self.n_inputs) and np.allclose(V[s], V[best], rtol=1e-11, atol=1e-11, equal_nan=True):
                    rep[s] = best
        return self._rewired(rep)

    def _rewired(self, rep: np.ndarray) -> Formula:
        """Read slot rep[s] wherever slot s was read; drop E's no longer used."""
        first = self.n_inputs + 1
        a, b, outputs = rep[self.a], rep[self.b], rep[self.outputs]
        used = np.zeros(self.n_slots, dtype=bool)
        used[outputs] = True
        for j in range(len(self) - 1, -1, -1):  # users come after their operands
            if used[first + j]:
                used[a[j]] = used[b[j]] = True
        keep = np.flatnonzero(used[first:])
        new = np.arange(self.n_slots)
        new[first + keep] = first + np.arange(len(keep))
        return Formula(new[a[keep]], new[b[keep]], new[outputs], self.n_inputs)


def _fingerprint(v: np.ndarray) -> bytes:
    """The first 24 values rounded to float32 (about 7 significant digits),
    so values that differ only by rounding noise share a key; -0 counts as 0."""
    v = v[:24]
    with np.errstate(over="ignore"):  # beyond float32's range is +-inf: still a fine key
        parts = [np.where(np.isnan(x), np.nan, x + 0.0).astype(np.float32) for x in (v.real, v.imag)]
    return b"".join(p.tobytes() for p in parts)
