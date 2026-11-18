import math

from eml_study.boundary_analysis import crossings, stripe_ratio


def test_eml_ratio_at_2_5():
    r = crossings("eml", 2.5)
    assert len(r) == 3
    assert abs(stripe_ratio(r) - 0.9989) < 1e-3


def test_eml_ratio_at_3():
    assert abs(stripe_ratio(crossings("eml", 3.0)) - 0.2875) < 1e-3


def test_eml_below_minimum_has_one_negative_crossing():
    r = crossings("eml", 2.0)
    assert len(r) == 1
    assert abs(r[0] - (-0.287)) < 2e-3


def test_cubic_ratio_at_zero():
    assert abs(stripe_ratio(crossings("cubic", 0.0)) - 1.0) < 1e-12


def test_perceptron_one_crossing():
    for t in (-3.0, 0.0, 2.5):
        assert len(crossings("perceptron", t)) == 1


def test_sine_roots_at_zero():
    r = crossings("sine", 0.0)
    assert len(r) == 13
    for k, root in zip(range(-6, 7), r):
        assert abs(root - k * math.pi) < 1e-12
