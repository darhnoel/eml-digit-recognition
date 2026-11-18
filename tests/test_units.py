import math

import torch

from eml_study.units import N_PARAMS, UNITS, guard_counts, logits

EML = UNITS["eml"]


def t(x):
    return torch.tensor(x, dtype=torch.float64)


def test_default_dtype_is_float64():
    assert torch.get_default_dtype() == torch.float64


def test_eml_minimum_value():
    assert abs(EML.g(t(0.5671432904)).item() - 2.3303661248) < 1e-9


def test_eml_at_minus_one_is_inverse_e():
    assert abs(EML.g(t(-1.0)).item() - math.exp(-1)) < 1e-10


def test_eml_increasing_on_negative_side():
    s = torch.linspace(-5, -1e-6, 2000)
    vals = EML.g(s)
    assert torch.all(vals[1:] > vals[:-1])


def test_pole_guard_fires_and_output_finite():
    s = t([[1e-15]])
    pole, clamp = guard_counts(EML, s)
    assert pole.tolist() == [1] and clamp.tolist() == [0]
    assert torch.isfinite(EML.g(s)).all()


def test_exp_clamp_fires_and_gradient_finite():
    s = t([[400.0]]).requires_grad_(True)
    pole, clamp = guard_counts(EML, s)
    assert clamp.tolist() == [1] and pole.tolist() == [0]
    out = EML.g(s)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(s.grad).all()


def test_non_eml_guard_counts_are_zero():
    pole, clamp = guard_counts(UNITS["cubic"], t([[0.0, 1e-15, 400.0]]))
    assert pole.tolist() == [0] and clamp.tolist() == [0]


def test_cubic_roots_at_t_zero():
    for r in (-1.0, 0.0, 1.0):
        assert abs(UNITS["cubic"].g(t(r)).item()) < 1e-15


def test_sine_at_half_pi():
    assert abs(UNITS["sine"].g(t(math.pi / 2)).item() - 1.0) < 1e-15


def test_shapes_and_finite_gradients_for_all_units():
    assert N_PARAMS == 5
    gen = torch.Generator().manual_seed(0)
    X = torch.rand(30, 2, generator=gen) * 2 - 1
    for unit in UNITS.values():
        params = torch.randn(4, N_PARAMS, generator=gen).requires_grad_(True)
        z = logits(params, X, unit)
        assert z.shape == (4, 30)
        z.sum().backward()
        assert torch.isfinite(params.grad).all(), unit.name


def test_eml_gprime_matches_formula():
    for s in (-1.0, 0.5, 2.0):
        expected = math.exp(s) - 1.0 / s
        assert abs(EML.gprime(t(s)).item() - expected) < 1e-12
