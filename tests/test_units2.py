import math

import torch

from eml_study.units2 import N_PARAMS2, PAIRS, g2, guard_counts2, logits2


def t(x):
    return torch.tensor(x, dtype=torch.float64)


def test_eml_two_input_value():
    v = g2(PAIRS["eml"], t(1.0), t(2.0)).item()
    assert abs(v - (math.e - math.log(2.0))) < 1e-12


def test_eml_funnel_half_width():
    # boundary e^{s1} - ln|s2| = t  ->  |s2| = exp(e^{s1} - t); at s1=0, t=2: e^{-1}
    s2 = math.exp(math.exp(0.0) - 2.0)
    assert abs(s2 - math.exp(-1)) < 1e-15
    assert abs(g2(PAIRS["eml"], t(0.0), t(s2)).item() - 2.0) < 1e-12
    assert abs(g2(PAIRS["eml"], t(0.0), t(-s2)).item() - 2.0) < 1e-12


def test_pair_values():
    s1, s2 = t(0.5), t(-0.3)
    expect = {
        "linear": 0.5 + 0.3, "parabola": 0.25 + 0.3, "ellipse": 0.25 + 0.09,
        "hyperbola": 0.25 - 0.09, "sine-wave": math.sin(0.5) + 0.3,
        "exp-half": math.exp(0.5) + 0.3, "log-half": 0.5 - math.log(0.3),
        "eml": math.exp(0.5) - math.log(0.3),
    }
    for name, e in expect.items():
        assert abs(g2(PAIRS[name], s1, s2).item() - e) < 1e-12, name


def test_guards():
    s1 = t([[400.0, 0.0]])
    s2 = t([[1.0, 1e-15]])
    pole, clamp = guard_counts2(PAIRS["eml"], s1, s2)
    assert pole.tolist() == [1] and clamp.tolist() == [1]
    pole, clamp = guard_counts2(PAIRS["parabola"], s1, s2)
    assert pole.tolist() == [0] and clamp.tolist() == [0]
    assert torch.isfinite(g2(PAIRS["eml"], s1, s2)).all()


def test_shapes_and_finite_gradients():
    gen = torch.Generator().manual_seed(0)
    X = torch.rand(30, 2, generator=gen) * 2 - 1
    for pair in PAIRS.values():
        params = torch.randn(4, N_PARAMS2, generator=gen).requires_grad_(True)
        z = logits2(params, X, pair)
        assert z.shape == (4, 30)
        z.sum().backward()
        assert torch.isfinite(params.grad).all(), pair.name
