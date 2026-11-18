"""Unit families: single-input (first study) and two-input (follow-up).

A Family bundles everything train.py and report.py need to know about one
study: its units, tasks, task roles, parameter layout and forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from eml_study import tasks, tasks2, units, units2


@dataclass(frozen=True)
class Family:
    name: str
    units: list[str]
    tasks: list[str]
    controls: set[str]
    evaluated: set[str]
    sanity: set[str]
    n_params: int
    make_task: Callable
    # forward(unit, params, X) -> (aux, z); aux is passed to guards/diag
    forward: Callable
    guards: Callable  # (unit, aux) -> (pole_hits, clamp_hits) per seed
    diag: Callable  # (unit, aux) -> (min_abs, max_metric) per seed, or None
    init_g: Callable  # (unit, w_row(tensor), X) -> g values for the median t
    g1_static: str | None = None


# ---- single-input -----------------------------------------------------------

def _fwd1(unit, params, X):
    act = units.UNITS[unit]
    s = units.pre_activation(params, X)
    return s, params[:, 4:5] * (params[:, 3:4] - act.g(s))


def _guards1(unit, s):
    return units.guard_counts(units.UNITS[unit], s)


def _diag1(unit, s):
    if unit != "eml":
        return None
    return s.abs().min(1).values, units.UNITS["eml"].gprime(s).abs().max(1).values


def _init_g1(unit, w, X):
    s = X[:, 0] * w[0] + X[:, 1] * w[1] + w[2]
    return units.UNITS[unit].g(s)


SINGLE = Family(
    name="single",
    units=list(units.UNITS),
    tasks=list(tasks.TASKS),
    controls=set(tasks.CONTROL_TASKS),
    evaluated=set(tasks.EVALUATED_TASKS),
    sanity=set(tasks.SANITY_TASKS),
    n_params=units.N_PARAMS,
    make_task=tasks.make_task,
    forward=_fwd1,
    guards=_guards1,
    diag=_diag1,
    init_g=_init_g1,
)


# ---- two-input --------------------------------------------------------------

def _fwd2(unit, params, X):
    pair = units2.PAIRS[unit]
    s1, s2 = units2.pre_activation2(params, X)
    return (s1, s2), params[:, 7:8] * (params[:, 6:7] - units2.g2(pair, s1, s2))


def _guards2(unit, aux):
    return units2.guard_counts2(units2.PAIRS[unit], *aux)


def _diag2(unit, aux):
    """For units with ln|s2|: min |s2| and max 1/|s2| (the ln-side pole factor)."""
    if not units2.PAIRS[unit].uses_log:
        return None
    a = torch.clamp(aux[1].abs(), min=units.EPS)
    return a.min(1).values, (1.0 / a).max(1).values


def _init_g2(unit, w, X):
    s1 = X[:, 0] * w[0] + X[:, 1] * w[1] + w[2]
    s2 = X[:, 0] * w[3] + X[:, 1] * w[4] + w[5]
    return units2.g2(units2.PAIRS[unit], s1, s2)


TWO = Family(
    name="two",
    units=list(units2.PAIRS),
    tasks=list(tasks2.TASKS2),
    controls=set(tasks2.CONTROL_TASKS2),
    evaluated=set(tasks2.EVALUATED_TASKS2),
    sanity=set(tasks2.SANITY_TASKS2),
    n_params=units2.N_PARAMS2,
    make_task=tasks2.make_task2,
    forward=_fwd2,
    guards=_guards2,
    diag=_diag2,
    init_g=_init_g2,
    g1_static=units2.G1_TABLE,
)

FAMILIES = {"single": SINGLE, "two": TWO}
