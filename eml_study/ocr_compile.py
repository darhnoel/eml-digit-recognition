"""Compile a trained digit classifier (OCR on 8x8 digits) into pure EML.

1. Train a 64 -> 16 -> 10 tanh MLP on scikit-learn's bundled 8x8 digits.
2. Round every weight and bias to a 9-bit-mantissa number n * 2^-k.
3. Compile the network into EML expressions whose only leaves are the
   constant 1 and the 64 raw pixel values (0..16): one expression per class
   score, sharing subexpressions.
4. Evaluate the EML expressions on held-out digits and compare with the
   ordinary float network.

Run: python -m eml_study.ocr_compile [--out-dir results/ocr]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits

import eml_study  # noqa: F401  (float64 default)
from eml_study.eml_compiler import Builder

HIDDEN = 16


def load_split(seed: int = 0):
    d = load_digits()
    X, y = d.data.astype(np.float64), d.target.astype(np.int64)
    perm = np.random.default_rng(seed).permutation(len(X))
    tr, te = perm[:1437], perm[1437:]
    return X[tr], y[tr], X[te], y[te]


def train_mlp(Xtr, ytr, hidden=HIDDEN, steps=2000, seed=0):
    torch.manual_seed(seed)
    X = torch.from_numpy(Xtr / 16.0)
    y = torch.from_numpy(ytr)
    W1 = (torch.randn(hidden, 64) * 0.3).requires_grad_(True)
    b1 = torch.zeros(hidden, requires_grad=True)
    W2 = (torch.randn(10, hidden) * 0.3).requires_grad_(True)
    b2 = torch.zeros(10, requires_grad=True)
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        logits = torch.tanh(X @ W1.T + b1) @ W2.T + b2
        torch.nn.functional.cross_entropy(logits, y).backward()
        opt.step()
    # fold the /16 input scaling into W1 so the expression reads raw pixels
    return (W1.detach().numpy() / 16.0, b1.detach().numpy(), W2.detach().numpy(), b2.detach().numpy())


def forward_np(params, X):
    W1, b1, W2, b2 = params
    return np.tanh(X @ W1.T + b1) @ W2.T + b2


def compile_mlp(params):
    """Return (builder, output roots, quantized params)."""
    W1, b1, W2, b2 = params
    B = Builder()
    q = [np.zeros_like(p) for p in params]

    def weighted_sum(ln_inputs, weights, bias, qw, qb):
        pos, neg = [], []
        for i, (lnx, w) in enumerate(zip(ln_inputs, weights)):
            if w == 0:
                continue
            lnw, qa = B.ln_abs_quantized(w)
            qw[i] = np.sign(w) * qa
            term = B.exp(B.add(lnw, lnx))  # |w| * x
            (pos if w > 0 else neg).append(term)
        if bias != 0:
            lnb, qa = B.ln_abs_quantized(bias)
            qb[...] = np.sign(bias) * qa
            (pos if bias > 0 else neg).append(B.exp(lnb))
        return B.sub(B.sum(pos), B.sum(neg))

    ln_pixels = [B.ln(B.var(i)) for i in range(W1.shape[1])]
    one, two = B.one, B.integer(2)
    ln_two = B.ln(two)
    ln_hidden = []
    for j in range(W1.shape[0]):
        pre = weighted_sum(ln_pixels, W1[j], b1[j], q[0][j], q[1][j : j + 1])
        e2 = B.exp(B.add(pre, pre))  # e^{2p}
        frac = B.exp(B.sub(ln_two, B.ln(B.add(e2, one))))  # 2 / (e^{2p} + 1)
        h = B.sub(one, frac)  # tanh p
        ln_hidden.append(B.ln(h))
    roots = [
        weighted_sum(ln_hidden, W2[k], b2[k], q[2][k], q[3][k : k + 1])
        for k in range(W2.shape[0])
    ]
    return B, roots, q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/ocr")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xte, yte = load_split()
    t0 = time.time()
    params = train_mlp(Xtr, ytr)
    acc_float = (forward_np(params, Xte).argmax(1) == yte).mean()

    t1 = time.time()
    B, roots, qparams = compile_mlp(params)
    ref_q = forward_np(qparams, Xte)
    acc_quant = (ref_q.argmax(1) == yte).mean()
    stats = B.stats(roots)
    per_class = [B.stats([r]) for r in roots]

    t2 = time.time()
    Z = B.evaluate(roots, Xte)
    t3 = time.time()
    acc_eml = (Z.real.argmax(1) == yte).mean()
    agree = (Z.real.argmax(1) == ref_q.argmax(1)).mean()
    summary = {
        "test_samples": int(len(yte)),
        "accuracy_float_network": float(acc_float),
        "accuracy_quantized_network": float(acc_quant),
        "accuracy_pure_eml": float(acc_eml),
        "eml_vs_quantized_prediction_agreement": float(agree),
        "max_abs_logit_error": float(np.abs(Z.real - ref_q).max()),
        "max_abs_imag_part": float(np.abs(Z.imag).max()),
        "non_finite_outputs": int((~np.isfinite(Z)).sum()),
        "dag_nodes": stats["dag_nodes"],
        "dag_eml_nodes": stats["dag_eml_nodes"],
        "expanded_tree_nodes_all_10_outputs": stats["tree_size_total"],
        "expanded_tree_nodes_one_output_min": min(s["tree_size_total"] for s in per_class),
        "expanded_tree_nodes_one_output_max": max(s["tree_size_total"] for s in per_class),
        "max_depth": stats["max_depth"],
        "network_parameters": int(sum(p.size for p in params)),
        "seconds_train": round(t1 - t0, 2),
        "seconds_compile": round(t2 - t1, 2),
        "seconds_eval_eml": round(t3 - t2, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Digit OCR compiled to pure EML", ""]
    lines += [f"- **{k}**: {v:,}" if isinstance(v, int) else f"- **{k}**: {v}" for k, v in summary.items()]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
