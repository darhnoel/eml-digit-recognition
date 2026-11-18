"""The seven single-unit activations and the shared unit.

Every unit computes s = w1*u1 + w2*u2 + b, then z = k * (t - g(s)).
Only the activation g differs between units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

import eml_study  # noqa: F401  (sets float64 default)

EPS = 1e-12
EXP_CLAMP = 350.0
N_PARAMS = 5

Tensor = torch.Tensor


@dataclass(frozen=True)
class Activation:
    name: str
    g: Callable[[Tensor], Tensor]
    gprime: Callable[[Tensor], Tensor]


def _eml_g(s: Tensor) -> Tensor:
    return torch.exp(torch.clamp(s, max=EXP_CLAMP)) - torch.log(
        torch.clamp(s.abs(), min=EPS)
    )


def _eml_gprime(s: Tensor) -> Tensor:
    sign = torch.where(s >= 0, torch.ones_like(s), -torch.ones_like(s))
    return torch.exp(torch.clamp(s, max=EXP_CLAMP)) - 1.0 / (
        sign * torch.clamp(s.abs(), min=EPS)
    )


UNITS: dict[str, Activation] = {
    "perceptron": Activation("perceptron", lambda s: s, lambda s: torch.ones_like(s)),
    "gaussian": Activation(
        "gaussian",
        lambda s: -torch.exp(-(s**2)),
        lambda s: 2 * s * torch.exp(-(s**2)),
    ),
    "quadratic": Activation("quadratic", lambda s: s**2, lambda s: 2 * s),
    "abs": Activation("abs", lambda s: s.abs(), lambda s: torch.sign(s)),
    "cubic": Activation("cubic", lambda s: s**3 - s, lambda s: 3 * s**2 - 1),
    "sine": Activation("sine", torch.sin, torch.cos),
    "eml": Activation("eml", _eml_g, _eml_gprime),
}


def guard_counts(activation: Activation, s: Tensor) -> tuple[Tensor, Tensor]:
    """Per-seed (row) counts of pole-guard hits and exp-clamp hits."""
    if activation.name != "eml":
        zeros = torch.zeros(s.shape[0], dtype=torch.long)
        return zeros, zeros.clone()
    pole = (s.detach().abs() < EPS).sum(dim=-1)
    clamp = (s.detach() > EXP_CLAMP).sum(dim=-1)
    return pole, clamp


def pre_activation(params: Tensor, X: Tensor) -> Tensor:
    """s for every seed and point, computed element by element (no matmul).

    params: (n_seeds, 5) ordered [w1, w2, b, t, k]; X: (n_points, 2).
    Returns (n_seeds, n_points).
    """
    return (
        X[:, 0][None, :] * params[:, 0:1]
        + X[:, 1][None, :] * params[:, 1:2]
        + params[:, 2:3]
    )


def logits(params: Tensor, X: Tensor, activation: Activation) -> Tensor:
    s = pre_activation(params, X)
    return params[:, 4:5] * (params[:, 3:4] - activation.g(s))
