"""Compute the G5 verdict and write report.md and plots.

Run: python -m eml_study.report [--input PATH] [--out-dir DIR]
                                [--plot-seed UNIT TASK LR SEED]
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from eml_study.family import FAMILIES, SINGLE, Family  # noqa: E402
from eml_study.train import LRS, RunRecord, params_of, read_csv  # noqa: E402
SCOPE_SENTENCE = (
    "This verdict concerns one real-valued, shared-sum EML unit. "
    "It does not decide two-input or multi-layer EML."
)
SCOPE_SENTENCE_TWO = (
    "This verdict concerns one real-valued, two-input EML unit, e^{s1} − ln|s2|, "
    "against seven two-input units of the same size. It does not decide multi-layer "
    "or complex-valued EML."
)
LIMITATIONS = (
    "**Limitation (t/k scale).** The threshold `t` and scale `k` work on different "
    "scales for different activations, so one Adam step moves some units' boundaries "
    "more than others'. Part of any reliability gap may come from parameter scale "
    "rather than from the activation's shape. Results at every learning rate are "
    "shown above so the effect is visible."
)
GO_GAP = 0.20
CLIP_SHARE, NAN_SHARE = 0.05, 0.01
POLE_WARN = 1e6


@dataclass
class Verdict:
    state: str
    rule: int
    tasks: list[str] = field(default_factory=list)


def _cells(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r.unit, r.task, r.lr)].append(r)
    return cells


def best_rates(rows) -> dict[tuple[str, str], float]:
    cells = _cells(rows)
    best = {}
    for unit, task in {(r.unit, r.task) for r in rows}:
        scored = []
        for lr in sorted({r.lr for r in rows if r.unit == unit and r.task == task}):
            runs = cells[(unit, task, lr)]
            frac = sum(r.train_acc >= 0.99 for r in runs) / len(runs)
            scored.append((-frac, lr))
        best[(unit, task)] = min(scored)[1]
    return best


def success_rates(rows, best) -> dict[tuple[str, str], float]:
    cells = _cells(rows)
    out = {}
    for (unit, task), lr in best.items():
        runs = cells[(unit, task, lr)]
        out[(unit, task)] = sum(r.success for r in runs) / len(runs)
    return out


def best_runs(rows, best, unit, task):
    return [r for r in rows if r.unit == unit and r.task == task and r.lr == best[(unit, task)]]


def _comparisons(fam: Family) -> list[str]:
    return [u for u in fam.units if u != "eml"]


def verdict(rows, fam: Family = SINGLE) -> Verdict:
    # Rule 1: any success on a control task, at any rate.
    bad = sorted({r.task for r in rows if r.task in fam.controls and r.success})
    if bad:
        return Verdict("INVALID", 1, bad)
    best = best_rates(rows)
    rate = success_rates(rows, best)
    # Rule 2: per (unit, evaluated task), on best-rate runs.
    flagged = []
    for unit in fam.units:
        for task in sorted(fam.evaluated):
            if (unit, task) not in best:
                continue
            runs = best_runs(rows, best, unit, task)
            clip = sum(r.clip_step_frac > 0.5 for r in runs) / len(runs)
            nan = sum(r.nan_flag for r in runs) / len(runs)
            if clip > CLIP_SHARE or nan > NAN_SHARE:
                flagged.append(f"{unit}:{task}")
    if flagged:
        return Verdict("INCONCLUSIVE", 2, flagged)
    # Rule 3: EML ahead of the best comparison unit by >= 20 points on any evaluated task.
    go = []
    for task in sorted(fam.evaluated):
        best_comp = max(rate.get((c, task), 0.0) for c in _comparisons(fam))
        if rate.get(("eml", task), 0.0) - best_comp >= GO_GAP - 1e-12:
            go.append(task)
    if go:
        return Verdict("GO", 3, go)
    return Verdict("NO-GO", 4, [])


def pole_warnings(rows) -> list[RunRecord]:
    best = best_rates(rows)
    return [
        r for r in rows
        if r.unit == "eml" and r.success and r.lr == best[(r.unit, r.task)]
        and r.max_abs_gprime is not None and r.max_abs_gprime > POLE_WARN
    ]


def _pct(x):
    return f"{100 * x:.0f}%"


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def render_report(rows, v: Verdict, g1_markdown: str, plot_links: list[str] | None = None,
                  fam: Family = SINGLE) -> str:
    best = best_rates(rows)
    rate = success_rates(rows, best)
    cells = _cells(rows)
    units = [u for u in fam.units if any(r.unit == u for r in rows)]
    tasks = [t for t in fam.tasks if any(r.task == t for r in rows)]
    lrs = sorted({r.lr for r in rows})
    n_seeds = max(len(c) for c in cells.values())
    title = "single-unit" if fam.name == "single" else "two-input"
    out = [f"# EML {title} study: report", ""]

    warned = pole_warnings(rows)
    metric = "max |g'(s)|" if fam.name == "single" else "max 1/|s2|"
    if warned:
        out += ["> **Pole warning** (does not change the verdict): best-rate EML successes "
                f"with final {metric} > {POLE_WARN:g}:", ""]
        for r in warned:
            out.append(f"> - {r.task} lr={r.lr:g} seed={r.seed}: {metric} = {r.max_abs_gprime:.3g}, "
                       f"min |s| = {r.min_abs_s:.3g} ([plot](plots/warn_{r.task}_{r.lr:g}_{r.seed}.png))")
        out.append("")

    desc = {
        "INVALID": "a control task had a successful run",
        "INCONCLUSIVE": "clipping or non-finite values exceeded the G5 rule 2 limits",
        "GO": "EML led the best comparison unit by at least 20 points",
        "NO-GO": "EML did not lead the best comparison unit by 20 points on any evaluated task",
    }[v.state]
    out += [f"## Verdict: **{v.state}** (G5 rule {v.rule})", "",
            f"{desc}" + (f": {', '.join(v.tasks)}." if v.tasks else "."), "",
            f"*{SCOPE_SENTENCE if fam.name == 'single' else SCOPE_SENTENCE_TWO}*", "",
            f"Seeds per (unit, task, rate): {n_seeds}. Learning rates: {', '.join(f'{x:g}' for x in lrs)}.", ""]

    out += ["## Success rate at each unit's best learning rate", "",
            "| unit | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for u in units:
        cells_txt = [f"{_pct(rate[(u, t)])} (lr {best[(u, t)]:g})" for t in tasks]
        out.append(f"| {u} | " + " | ".join(cells_txt) + " |")
    out.append("")

    out += ["## Gap: EML minus best comparison unit (evaluated tasks)", "",
            "| task | EML | best comparison | gap |", "|---|---|---|---|"]
    for t in [t for t in tasks if t in fam.evaluated]:
        bc = max(_comparisons(fam), key=lambda c: rate.get((c, t), 0.0))
        out.append(f"| {t} | {_pct(rate[('eml', t)])} | {bc} {_pct(rate[(bc, t)])} | "
                   f"{100 * (rate[('eml', t)] - rate[(bc, t)]):+.0f} pts |")
    out.append("")

    out += ["## Success rate at every learning rate", "",
            "| unit | task | " + " | ".join(f"lr {x:g}" for x in lrs) + " |",
            "|---|---|" + "---|" * len(lrs)]
    for u in units:
        for t in tasks:
            vals = []
            for lr in lrs:
                runs = cells.get((u, t, lr), [])
                vals.append(_pct(sum(r.success for r in runs) / len(runs)) if runs else "–")
            out.append(f"| {u} | {t} | " + " | ".join(vals) + " |")
    out.append("")

    out += ["## Median train / test accuracy", "",
            "| unit | task | " + " | ".join(f"lr {x:g}" for x in lrs) + " |",
            "|---|---|" + "---|" * len(lrs)]
    for u in units:
        for t in tasks:
            vals = []
            for lr in lrs:
                runs = cells.get((u, t, lr), [])
                vals.append(f"{_median([r.train_acc for r in runs]):.3f} / {_median([r.test_acc for r in runs]):.3f}"
                            if runs else "–")
            out.append(f"| {u} | {t} | " + " | ".join(vals) + " |")
    out.append("")

    out += ["## Stability (best-rate runs)", "",
            "| unit | task | NaN share | share clipped >50% of steps | median clip_step_frac | "
            + ("EML median pole_guard_frac | EML exp_clamp_hits | EML min |s| | EML max |g'| |" if fam.name == "single"
             else "EML median pole_guard_frac | EML exp_clamp_hits | EML min |s2| | EML max 1/|s2| |"),
            "|---|---|---|---|---|---|---|---|---|"]
    for u in units:
        for t in tasks:
            runs = best_runs(rows, best, u, t)
            nan = sum(r.nan_flag for r in runs) / len(runs)
            clip = sum(r.clip_step_frac > 0.5 for r in runs) / len(runs)
            mclip = _median([r.clip_step_frac for r in runs])
            if u == "eml":
                extra = (f"{_median([r.pole_guard_frac for r in runs]):.2g} | "
                         f"{sum(r.exp_clamp_hits for r in runs)} | "
                         f"{min(r.min_abs_s for r in runs):.3g} | "
                         f"{max(r.max_abs_gprime for r in runs):.3g}")
            else:
                extra = "– | – | – | –"
            out.append(f"| {u} | {t} | {_pct(nan)} | {_pct(clip)} | {mclip:.3f} | {extra} |")
    out.append("")

    out += ["## G1: boundary families", "", g1_markdown.strip(), "", LIMITATIONS, ""]
    if plot_links:
        out += ["## Plots", ""] + [f"- [{p}]({p})" for p in plot_links] + [""]
    return "\n".join(out)


def plot_run(r: RunRecord, path: Path, title: str | None = None, fam: Family = SINGLE) -> None:
    X, y = fam.make_task(r.task, "test")
    g = np.linspace(-1, 1, 200)
    gx, gy = np.meshgrid(g, g)
    grid = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], 1))
    params = torch.tensor([params_of(r)])
    with torch.no_grad():
        z = fam.forward(r.unit, params, grid)[1][0].numpy().reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.contourf(gx, gy, (z > 0).astype(float), levels=[-0.5, 0.5, 1.5], colors=["#f2c6b4", "#b4d4f2"])
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=6, c="#1f5fa8", label="A")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=6, c="#b5452a", label="B")
    ax.set_title(title or f"{r.unit} {r.task} lr={r.lr:g} seed={r.seed}\n"
                 f"train {r.train_acc:.3f} test {r.test_acc:.3f}", fontsize=8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def make_plots(rows, out_dir: Path, fam: Family = SINGLE) -> list[str]:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    best = best_rates(rows)
    links = []
    for (u, t), lr in sorted(best.items()):
        runs = sorted(best_runs(rows, best, u, t), key=lambda r: (-r.test_acc, r.final_loss, r.seed))
        for tag, r in (("best", runs[0]), ("worst", runs[-1])):
            name = f"{u}_{t}_{tag}.png"
            plot_run(r, plots / name, fam=fam)
            links.append(f"plots/{name}")
    for r in pole_warnings(rows):
        name = f"warn_{r.task}_{r.lr:g}_{r.seed}.png"
        plot_run(r, plots / name, fam=fam)
        links.append(f"plots/{name}")
    # bar chart of best-rate success rates
    rate = success_rates(rows, best)
    units = [u for u in fam.units if any(r.unit == u for r in rows)]
    tasks = [t for t in fam.tasks if any(r.task == t for r in rows)]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(units)
    for i, u in enumerate(units):
        ax.bar(np.arange(len(tasks)) + i * w, [rate[(u, t)] for t in tasks], w, label=u)
    ax.set_xticks(np.arange(len(tasks)) + 0.4 - w / 2, tasks)
    ax.set_ylabel("success rate (best lr)")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(plots / "success_rates.png", dpi=100)
    plt.close(fig)
    links.append("plots/success_rates.png")
    return links


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=sorted(FAMILIES), default="single")
    ap.add_argument("--input", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--plot-seed", nargs=4, metavar=("UNIT", "TASK", "LR", "SEED"))
    args = ap.parse_args(argv)
    fam = FAMILIES[args.family]
    default_dir = "results" if fam.name == "single" else "results/two_input"
    out = Path(args.out_dir or default_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.input or f"{default_dir}/results.csv")
    if args.plot_seed:
        u, t, lr, seed = args.plot_seed
        match = [r for r in rows if r.unit == u and r.task == t and r.lr == float(lr) and r.seed == int(seed)]
        if not match:
            raise SystemExit("no such run")
        (out / "plots").mkdir(exist_ok=True)
        path = out / "plots" / f"seed_{u}_{t}_{lr}_{seed}.png"
        plot_run(match[0], path, fam=fam)
        print(f"wrote {path}")
        return
    v = verdict(rows, fam)
    if fam.g1_static is not None:
        g1 = fam.g1_static
    else:
        g1_path = out / "g1_summary.md"
        g1 = g1_path.read_text() if g1_path.exists() else "(g1_summary.md not found; run boundary_analysis first)"
    links = make_plots(rows, out, fam)
    (out / "report.md").write_text(render_report(rows, v, g1, links, fam))
    print(f"verdict: {v.state} (rule {v.rule}) {', '.join(v.tasks)}")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
