"""Two-input units: z = k * (t - (g_a(s1) - g_b(s2))), 8 parameters.

params order: [a1, a2, b, c1, c2, d, t, k]
s1 = a1*u1 + a2*u2 + b, s2 = c1*u1 + c2*u2 + d.
See docs/designs/eml_two_input_study.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

import eml_study  # noqa: F401  (float64 default)
from eml_study.units import EPS, EXP_CLAMP

N_PARAMS2 = 8
Tensor = torch.Tensor


def _lin(s):
    return s


def _sq(s):
    return s**2


def _negsq(s):
    return -(s**2)


def _exp(s):
    return torch.exp(torch.clamp(s, max=EXP_CLAMP))


def _logabs(s):
    return torch.log(torch.clamp(s.abs(), min=EPS))


@dataclass(frozen=True)
class Pair:
    name: str
    ga: Callable[[Tensor], Tensor]
    gb: Callable[[Tensor], Tensor]
    uses_exp: bool = False
    uses_log: bool = False


PAIRS: dict[str, Pair] = {
    "linear": Pair("linear", _lin, _lin),
    "parabola": Pair("parabola", _sq, _lin),
    "ellipse": Pair("ellipse", _sq, _negsq),
    "hyperbola": Pair("hyperbola", _sq, _sq),
    "sine-wave": Pair("sine-wave", torch.sin, _lin),
    "exp-half": Pair("exp-half", _exp, _lin, uses_exp=True),
    "log-half": Pair("log-half", _lin, _logabs, uses_log=True),
    "eml": Pair("eml", _exp, _logabs, uses_exp=True, uses_log=True),
}

G1_TABLE = """| unit | g_a(s1) | g_b(s2) | boundary in (s1, s2) |
|---|---|---|---|
| linear | s1 | s2 | straight line |
| parabola | s1² | s2 | parabola |
| ellipse | s1² | −s2² | ellipse (closed) |
| hyperbola | s1² | s2² | hyperbola; two crossing lines at t = 0 |
| sine-wave | sin s1 | s2 | wavy curve |
| exp-half | e^s1 | s2 | exponential curve |
| log-half | s1 | ln\\|s2\\| | funnel, half-width e^(s1−t) |
| eml | e^s1 | ln\\|s2\\| | funnel, half-width exp(e^s1 − t) |
"""


def pre_activation2(params: Tensor, X: Tensor) -> tuple[Tensor, Tensor]:
    x0, x1 = X[:, 0][None, :], X[:, 1][None, :]
    s1 = x0 * params[:, 0:1] + x1 * params[:, 1:2] + params[:, 2:3]
    s2 = x0 * params[:, 3:4] + x1 * params[:, 4:5] + params[:, 5:6]
    return s1, s2


def g2(pair: Pair, s1: Tensor, s2: Tensor) -> Tensor:
    return pair.ga(s1) - pair.gb(s2)


def logits2(params: Tensor, X: Tensor, pair: Pair) -> Tensor:
    s1, s2 = pre_activation2(params, X)
    return params[:, 7:8] * (params[:, 6:7] - g2(pair, s1, s2))


def guard_counts2(pair: Pair, s1: Tensor, s2: Tensor) -> tuple[Tensor, Tensor]:
    n = s1.shape[0]
    pole = (s2.detach().abs() < EPS).sum(-1) if pair.uses_log else torch.zeros(n, dtype=torch.long)
    clamp = (s1.detach() > EXP_CLAMP).sum(-1) if pair.uses_exp else torch.zeros(n, dtype=torch.long)
    return pole, clamp
