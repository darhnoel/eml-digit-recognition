"""Figure 4: size against accuracy of pure EML expressions for MNIST (Section 6).

Networks with 16, 32 and 64 hidden units, either pruned step by step
(compact compiler) or with a ternary first layer (ternary compiler). Every
expression is evaluated as an expression on all 10,000 test images.
Writes paper/data/frontier.json and paper/figures/frontier.pdf (a few minutes).

Run from the project root:  uv run python paper/frontier.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eml_study.compact_compile import compile_compact, compile_ternary  # noqa: E402
from eml_study.mnist_data import load  # noqa: E402
from eml_study.networks import prune_gradually, train_ternary  # noqa: E402

ROOT = Path(__file__).resolve().parent
HIDDEN = (16, 32, 64)
TERNARY_KEEPS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.03)


def measure():
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    rows = []

    def add(family, hidden, keep, compiled):
        formula, net = compiled
        scores = formula.scores(Xte)
        rows.append({
            "family": family, "hidden": hidden, "keep": keep, "weights": net.n_weights,
            "E_nodes": len(formula), "depth": formula.depth,
            "acc_E": float((scores.argmax(1) == yte).mean()),
            "same_as_network": float((scores.argmax(1) == net(Xte).argmax(1)).mean()),
            "max_logit_err": float(np.nanmax(np.abs(scores - net(Xte)))),
        })
        print(json.dumps(rows[-1]), flush=True)

    for hidden in HIDDEN:
        for keep, net in prune_gradually(Xtr, ytr, hidden):
            add("pruned", hidden, keep, compile_compact(net))
        for keep in TERNARY_KEEPS:
            add("ternary", hidden, keep, compile_ternary(train_ternary(Xtr, ytr, hidden, keep)))
    return rows


def plot(rows, merged):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for hidden, color in zip(HIDDEN, ("#a9a9a9", "#5c5c5c", "#000000")):
        for family, marker, ls in (("pruned", "o", "-"), ("ternary", "s", ":")):
            pts = sorted((r["E_nodes"], 100 * r["acc_E"]) for r in rows if (r["family"], r["hidden"]) == (family, hidden))
            ax.plot(*zip(*pts), marker=marker, ms=2.8, lw=0.8, ls=ls, color=color, label=f"{hidden} hidden, {family}")
    ax.plot([258842], [96.35], marker="*", ms=7, color="#000000", ls="none")
    ax.annotate("original (Section 5)", (258842, 96.35), xytext=(-8, -12), textcoords="offset points",
                ha="right", fontsize=8)
    ax.plot([merged["E_merged"]], [100 * merged["accuracy"]], marker="D", ms=4, mfc="white", mec="#000000", ls="none")
    ax.annotate("merged (Table 4)", (merged["E_merged"], 100 * merged["accuracy"]), xytext=(-6, 6),
                textcoords="offset points", ha="right", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"distinct $\mathrm{E}$ nodes (log scale)")
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_ylim(85, 98)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower right")
    fig.savefig(ROOT / "figures/frontier.pdf")


if __name__ == "__main__":
    data = ROOT / "data/frontier.json"
    rows = measure()
    data.write_text(json.dumps(rows, indent=2) + "\n")
    merged = json.loads((ROOT.parent / "results/compact/summary.json").read_text())
    plot(rows, merged)
