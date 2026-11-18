"""G1: decision boundaries each unit can draw, from the crossings of g(s) = t.

Uses unguarded g in plain float arithmetic and bisection.
Run: python -m eml_study.boundary_analysis [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable

OMEGA = 0.5671432904097838
S_RANGE = (-20.0, 20.0)

G: dict[str, Callable[[float], float]] = {
    "perceptron": lambda s: s,
    "gaussian": lambda s: -math.exp(-s * s),
    "quadratic": lambda s: s * s,
    "abs": abs,
    "cubic": lambda s: s**3 - s,
    "sine": math.sin,
    "eml": lambda s: math.exp(s) - math.log(abs(s)),
}

T_GRID: dict[str, list[float]] = {
    "eml": [2.34, 2.5, 3.0, 5.0, 10.0, 20.0],
    "cubic": [-0.38, -0.2, 0.0, 0.2, 0.38],
    "sine": [-0.9, 0.0, 0.9],
    "perceptron": [0.0],
    "gaussian": [-0.5],
    "quadratic": [0.25, 0.5],
    "abs": [0.25, 0.5],
}

PATTERN = {
    "perceptron": "A | B",
    "gaussian": "B | A | B",
    "quadratic": "B | A | B",
    "abs": "B | A | B",
    "cubic": "A | B | A | B",
    "sine": "A | B | A | B | ...",
    "eml": "A | B | A | B",
}


def _bisect(f: Callable[[float], float], a: float, b: float, tol: float = 1e-14):
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if (fa < 0) == (fb < 0):
        return None
    for _ in range(400):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0 or (b - a) < tol:
            return m
        if (fm < 0) == (fa < 0):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def _brackets(unit: str, s_range: tuple[float, float]) -> list[tuple[float, float]]:
    lo, hi = s_range
    if unit == "eml":
        return [(lo, -1e-300), (1e-300, OMEGA), (OMEGA, hi)]
    if unit == "cubic":
        c = 1 / math.sqrt(3)
        return [(lo, -c), (-c, c), (c, hi)]
    if unit == "sine":
        out = []
        j = math.floor((lo - math.pi / 2) / math.pi)
        while -math.pi / 2 + j * math.pi < hi:
            a = max(lo, -math.pi / 2 + j * math.pi)
            b = min(hi, math.pi / 2 + j * math.pi)
            if a < b:
                out.append((a, b))
            j += 1
        return out
    raise ValueError(unit)


def crossings(unit: str, t: float, s_range=S_RANGE) -> list[float]:
    """Every s in s_range where g(s) = t, sorted."""
    lo, hi = s_range
    if unit == "perceptron":
        roots = [t]
    elif unit == "gaussian":
        roots = [] if not (-1 < t < 0) else [-math.sqrt(-math.log(-t)), math.sqrt(-math.log(-t))]
    elif unit == "quadratic":
        roots = [] if t <= 0 else [-math.sqrt(t), math.sqrt(t)]
    elif unit == "abs":
        roots = [] if t <= 0 else [-t, t]
    else:
        f = lambda s: G[unit](s) - t  # noqa: E731
        roots = []
        for a, b in _brackets(unit, s_range):
            r = _bisect(f, a, b)
            if r is not None and not any(abs(r - q) < 1e-12 for q in roots):
                roots.append(r)
    return sorted(r for r in roots if lo <= r <= hi)


def stripe_ratio(roots: list[float]) -> float | None:
    """(s1 - s0) / (s2 - s1) for three consecutive roots (the middle three)."""
    if len(roots) < 3:
        return None
    i = (len(roots) - 3) // 2
    s0, s1, s2 = roots[i : i + 3]
    return (s1 - s0) / (s2 - s1)


def analyse() -> list[dict]:
    rows = []
    for unit, grid in T_GRID.items():
        for t in grid:
            roots = crossings(unit, t)
            ratio = stripe_ratio(roots) if (unit != "sine" or len(roots) >= 3) else None
            three = roots[(len(roots) - 3) // 2 :][:3] if len(roots) >= 3 else []
            rows.append(
                {
                    "unit": unit,
                    "t": t,
                    "n_roots": len(roots),
                    "s0": three[0] if three else "",
                    "s1": three[1] if three else "",
                    "s2": three[2] if three else "",
                    "ratio": ratio if ratio is not None else "",
                }
            )
    return rows


def summary_markdown(rows: list[dict]) -> str:
    lines = [
        "| unit | most lines | region pattern | width ratios seen |",
        "|---|---|---|---|",
    ]
    for unit in T_GRID:
        mine = [r for r in rows if r["unit"] == unit]
        most = max(r["n_roots"] for r in mine)
        most_txt = f"unlimited ({most} in s∈(−20, 20))" if unit == "sine" else str(most)
        ratios = [r["ratio"] for r in mine if r["ratio"] != ""]
        ratio_txt = f"{min(ratios):.3g} – {max(ratios):.3g}" if ratios else "n/a"
        pattern = PATTERN[unit].replace("|", "\\|")
        lines.append(f"| {unit} | {most_txt} | {pattern} | {ratio_txt} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = analyse()
    with open(out / "boundaries.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["unit", "t", "n_roots", "s0", "s1", "s2", "ratio"])
        w.writeheader()
        w.writerows(rows)
    (out / "g1_summary.md").write_text(summary_markdown(rows))
    print(f"wrote {out / 'boundaries.csv'} and {out / 'g1_summary.md'}")


if __name__ == "__main__":
    main()
