"""Small tanh networks for MNIST (784 -> hidden -> 10), trained in float64.

The input scaling 1/255 is folded into W1, so a network reads raw pixel
values 0..255, exactly like the formula compiled from it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

F64 = torch.float64
KEEPS = (0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02)  # gradual pruning schedule


class MLP(NamedTuple):
    W1: np.ndarray  # (hidden, 784), includes the 1/255 input scaling
    b1: np.ndarray
    W2: np.ndarray  # (10, hidden)
    b2: np.ndarray

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Logits for rows of raw pixels X (n, 784)."""
        return np.tanh(X @ self.W1.T + self.b1) @ self.W2.T + self.b2

    @property
    def n_weights(self) -> int:
        """Nonzero weights, biases not counted."""
        return int(np.count_nonzero(self.W1) + np.count_nonzero(self.W2))

    def save(self, path: Path) -> None:
        np.savez(path, **self._asdict())

    @classmethod
    def load(cls, path: Path) -> MLP:
        with np.load(path) as d:
            return cls(*(d[k] for k in cls._fields))


def train(X, y, hidden=32, epochs=15, *, init: MLP | None = None, mask=None, seed=0, lr=2e-3,
          bits: int | None = None, mask2=None) -> MLP:
    """Train from scratch, or continue from `init`; where `mask` is 0, W1 stays 0
    (and where `mask2` is 0, W2 stays 0).

    With `bits`, the forward pass uses every weight rounded to `bits`
    significant bits, as the compiler stores it (gradients pass straight
    through the rounding), so the network learns to work with coarse weights.
    """
    params = _init(hidden, seed) if init is None else [torch.tensor(p) for p in init]
    if init is not None:
        params[0] = params[0] * 255.0  # unfold the input scaling
    params = [p.clone().requires_grad_(True) for p in params]
    M = torch.ones_like(params[0]) if mask is None else torch.from_numpy(mask).double()
    M2 = None if mask2 is None else torch.from_numpy(mask2).double()
    with torch.no_grad():
        params[0].mul_(M)
        if M2 is not None:
            params[2].mul_(M2)

    def use(ps):
        W1, b1, W2, b2 = ps[0] * M, ps[1], ps[2] if M2 is None else ps[2] * M2, ps[3]
        if bits is None:
            return [W1, b1, W2, b2]
        # W1 is stored divided by 255: round the weight the formula holds
        return [round_ste(W1, bits, 255.0), *(round_ste(p, bits) for p in (b1, W2, b2))]

    _fit(params, use, X, y, epochs, seed, lr, weight_decay=1e-5)
    with torch.no_grad():
        params[0].mul_(M)
        if M2 is not None:
            params[2].mul_(M2)
    W1, b1, W2, b2 = (p.detach().numpy() for p in params)
    return MLP(W1 / 255.0, b1, W2, b2)


def train_ternary(X, y, hidden, keep, epochs=15, seed=0, lr=2e-3) -> MLP:
    """Layer 1 ternary per row: alpha_j times -1, 0 or +1, nonzero for the
    largest `keep` fraction of the row. Straight-through training."""

    def ternarize(W):
        mag = W.detach().abs()
        mask = (mag >= torch.quantile(mag, 1 - keep, dim=1, keepdim=True)).double()
        alpha = (mag * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1)
        return W + (alpha * torch.sign(W.detach()) * mask - W).detach()

    params = [p.requires_grad_(True) for p in _init(hidden, seed)]
    _fit(params, lambda ps: [ternarize(ps[0]), *ps[1:]], X, y, epochs, seed, lr, weight_decay=0)
    with torch.no_grad():
        W1 = ternarize(params[0]).numpy()
    b1, W2, b2 = (p.detach().numpy() for p in params[1:])
    return MLP(W1 / 255.0, b1, W2, b2)


def round_ste(W: torch.Tensor, bits: int, scale: float = 1.0) -> torch.Tensor:
    """W/scale rounded to `bits` significant bits (n 2^-k, 2^(bits-1) <= n < 2^bits),
    times scale; zeros stay zero. The gradient passes straight through."""
    w = W.detach() / scale
    k = bits - 1 - torch.floor(torch.log2(w.abs().clamp(min=1e-300)))
    q = torch.where(w == 0, 0.0, torch.sign(w) * torch.round(w.abs() * 2.0**k) * 2.0**-k) * scale
    return W + (q - W).detach()


def magnitude_mask(W: np.ndarray, keep: float) -> np.ndarray:
    """1 for the largest `keep` fraction of |W| (zeros stay pruned), else 0."""
    mag = np.abs(W)
    return ((mag >= np.quantile(mag, 1 - keep)) & (mag > 0)).astype(float)


def saliency_mask(W1: np.ndarray, spread: np.ndarray, n_keep: int) -> np.ndarray:
    """1 for the `n_keep` weights of W1 with the largest |w| * spread of their
    pixel (a weight on a pixel that hardly varies matters little), else 0."""
    score = np.where(W1 != 0, np.abs(W1) * spread[None, :], -1.0)
    threshold = np.sort(score[W1 != 0])[::-1][n_keep - 1]
    return ((score >= threshold) & (W1 != 0)).astype(float)


def prune_gradually(X, y, hidden, keeps=KEEPS, seed=0) -> Iterator[tuple[float, MLP]]:
    """Train, then over and over prune layer 1 a little further and retrain."""
    net = train(X, y, hidden, seed=seed)
    for keep in keeps:
        net = train(X, y, hidden, epochs=4, init=net, mask=magnitude_mask(net.W1, keep), seed=seed)
        yield keep, net


def _init(hidden: int, seed: int) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    return [
        torch.randn(hidden, 784, dtype=F64) / 28,
        torch.zeros(hidden, dtype=F64),
        torch.randn(10, hidden, dtype=F64) / hidden**0.5,
        torch.zeros(10, dtype=F64),
    ]


def _fit(params, use: Callable, X, y, epochs, seed, lr, weight_decay) -> None:
    """Adam on cross-entropy, batches of 128; use(params) gives the weights in use."""
    Xt, yt = torch.from_numpy(X / 255.0), torch.from_numpy(y)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xt), generator=gen).split(128):
            W1, b1, W2, b2 = use(params)
            opt.zero_grad()
            logits = torch.tanh(Xt[idx] @ W1.T + b1) @ W2.T + b2
            torch.nn.functional.cross_entropy(logits, yt[idx]).backward()
            opt.step()
