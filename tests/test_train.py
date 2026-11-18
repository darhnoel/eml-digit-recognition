import math

import torch

from eml_study.tasks import make_task
from eml_study.train import init_params, read_csv, train_batch, write_csv
from eml_study.units import UNITS


def _final(rec):
    return torch.tensor([rec.w1, rec.w2, rec.b, rec.t, rec.k])


def test_seed_independence_batch_vs_alone():
    for unit in ("eml", "cubic", "sine"):
        batch = train_batch(unit, "T4_stripes", 1e-2, [0, 1, 2, 3], steps=100)
        for i, seed in enumerate([0, 1, 2, 3]):
            alone = train_batch(unit, "T4_stripes", 1e-2, [seed], steps=100)[0]
            assert torch.allclose(_final(batch[i]), _final(alone), atol=1e-10, rtol=0), (unit, seed)


def test_perceptron_smoke_on_t1():
    recs = train_batch("perceptron", "T1_linear", 1e-1, range(10), steps=1000)
    assert sum(r.success for r in recs) >= 8


def test_nan_masking_freezes_only_the_bad_seed():
    X, _ = make_task("T1_linear", "train")
    init = init_params(UNITS["perceptron"], torch.from_numpy(X), [0, 1, 2])
    init[1, 4] = math.inf
    recs = train_batch("perceptron", "T1_linear", 1e-2, [0, 1, 2], steps=50, init=init)
    assert recs[1].nan_flag and not recs[0].nan_flag and not recs[2].nan_flag
    assert torch.equal(_final(recs[1]).nan_to_num(), init[1].nan_to_num())
    assert not torch.allclose(_final(recs[0]), init[0])
    assert not torch.allclose(_final(recs[2]), init[2])


def test_clip_counting():
    recs = train_batch("cubic", "T2_xor", 1e-2, [0, 1], steps=20, clip_norm=1e-6)
    assert all(r.clip_step_frac == 1.0 for r in recs)


def test_csv_roundtrip(tmp_path):
    recs = train_batch("eml", "T3_band", 1e-2, [0, 1], steps=10)
    recs += train_batch("sine", "T3_band", 1e-2, [0], steps=10)
    write_csv(recs, tmp_path / "r.csv")
    back = read_csv(tmp_path / "r.csv")
    assert back == recs


def test_two_input_seed_independence_and_params():
    batch = train_batch("eml", "U3_disc", 1e-2, [0, 1, 2], steps=50, family="two")
    for i, seed in enumerate([0, 1, 2]):
        alone = train_batch("eml", "U3_disc", 1e-2, [seed], steps=50, family="two")[0]
        assert batch[i] == alone
    r = batch[0]
    assert r.c1 is not None and r.min_abs_s is not None
    lin = train_batch("linear", "U1_linear", 1e-1, [0, 1], steps=500, family="two")
    assert all(x.success for x in lin) and lin[0].min_abs_s is None


def test_two_input_csv_roundtrip(tmp_path):
    recs = train_batch("ellipse", "U3_disc", 1e-2, [0, 1], steps=10, family="two")
    write_csv(recs, tmp_path / "r.csv")
    assert read_csv(tmp_path / "r.csv") == recs
