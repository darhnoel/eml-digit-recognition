import numpy as np
from eml_study.eml_compiler import Builder

from eml_study.formula import Formula

RNG = np.random.default_rng(0)
X = RNG.integers(0, 256, (50, 3)).astype(float)


def small() -> tuple[Builder, list[int]]:
    """x0 * x1 - x2 and e^x0 over three inputs."""
    B = Builder()
    x0, x1, x2 = (B.var(i) for i in range(3))
    return B, [B.sub(B.mul(x0, x1), x2), B.exp(x0)]


def test_scores_match_the_builder():
    B, roots = small()
    f = Formula.from_builder(B, roots, n_inputs=3)
    assert np.array_equal(f.scores(X), B.evaluate(roots, X).real)


def test_write_then_read_gives_the_same_formula(tmp_path):
    f = Formula.from_builder(*small(), n_inputs=3)
    f.write(tmp_path / "f.txt")
    g = Formula.read(tmp_path / "f.txt", n_inputs=3)
    assert all(np.array_equal(u, v) for u, v in zip((f.a, f.b, f.outputs), (g.a, g.b, g.outputs)))


def test_depth_and_size():
    B = Builder()
    f = Formula.from_builder(B, [B.exp(B.exp(B.var(0)))], n_inputs=1)  # E(E(p1, 1), 1)
    assert (len(f), f.depth, f.tree_sizes()[-1]) == (2, 2, 5)


def test_merge_equal_removes_a_detour_and_keeps_the_values():
    B = Builder()
    x = B.var(0)
    detour = B.exp(B.ln(x))  # equals x, written with 4 E's
    f = Formula.from_builder(B, [B.add(detour, B.exp(x))], n_inputs=1)
    merged = f.merge_equal(RNG.uniform(-5, 5, (64, 1)))
    assert len(merged) < len(f)
    Y = RNG.uniform(-5, 5, (20, 1))
    assert np.allclose(merged.scores(Y), f.scores(Y), rtol=1e-12)
