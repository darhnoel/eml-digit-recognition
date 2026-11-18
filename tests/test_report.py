from eml_study.report import (
    SCOPE_SENTENCE,
    best_rates,
    render_report,
    verdict,
)
from eml_study.tasks import TASKS
from eml_study.train import LRS, RunRecord
from eml_study.units import UNITS


def _rec(unit, task, lr, seed, success):
    return RunRecord(
        unit=unit, task=task, lr=lr, seed=seed,
        train_acc=1.0 if success else 0.5, test_acc=1.0 if success else 0.5,
        success=success, final_loss=0.1, nan_flag=False, pole_guard_frac=0.0,
        exp_clamp_hits=0, clip_step_frac=0.0,
        min_abs_s=0.1 if unit == "eml" else None,
        max_abs_gprime=10.0 if unit == "eml" else None,
        w1=1.0, w2=0.0, b=0.0, t=0.0, k=1.0,
    )


def make_rows(n_seeds=10, default=0.5):
    """Full 7 x 6 x 3 grid. Non-control cells: `default` share succeed; controls: none."""
    rows = []
    for u in UNITS:
        for t in TASKS:
            for lr in LRS:
                k = 0 if t in ("T5_circles", "T6_moons") else round(default * n_seeds)
                rows += [_rec(u, t, lr, s, s < k) for s in range(n_seeds)]
    return rows


def set_cell(rows, unit, task, lr=None, n_success=None, **fields):
    """Override cells. n_success sets the first n seeds successful (rest failed).
    Other fields apply to the first `count` seeds (default all) of the cell."""
    count = fields.pop("count", None)
    cell = [r for r in rows if r.unit == unit and r.task == task and (lr is None or r.lr == lr)]
    by_lr = {}
    for r in cell:
        by_lr.setdefault(r.lr, []).append(r)
    for runs in by_lr.values():
        runs.sort(key=lambda r: r.seed)
        if n_success is not None:
            for i, r in enumerate(runs):
                ok = i < n_success
                r.success, r.train_acc, r.test_acc = ok, (1.0 if ok else 0.5), (1.0 if ok else 0.5)
        for i, r in enumerate(runs):
            if count is None or i < count:
                for k, v in fields.items():
                    setattr(r, k, v)


def test_go_on_t4_only():
    rows = make_rows(n_seeds=20)  # everyone at 10/20 = 50%
    set_cell(rows, "eml", "T4_stripes", n_success=15)  # 75%: +25 pts
    v = verdict(rows)
    assert (v.state, v.rule, v.tasks) == ("GO", 3, ["T4_stripes"])


def test_no_go_when_lead_under_20_and_t1_ignored():
    rows = make_rows(n_seeds=20)
    for t in ("T2_xor", "T3_band", "T4_stripes"):
        set_cell(rows, "eml", t, n_success=13)  # 65%: +15 pts
    set_cell(rows, "eml", "T1_linear", n_success=18)  # +40 pts on sanity task
    assert verdict(rows).state == "NO-GO"


def test_invalid_on_control_success_even_at_non_best_rate():
    rows = make_rows(n_seeds=20)
    set_cell(rows, "eml", "T4_stripes", n_success=15)  # would be GO
    set_cell(rows, "quadratic", "T5_circles", lr=1e-3, n_success=1)
    v = verdict(rows)
    assert (v.state, v.rule) == ("INVALID", 1)


def test_inconclusive_on_comparison_unit_clipping():
    rows = make_rows(n_seeds=50)
    set_cell(rows, "cubic", "T3_band", clip_step_frac=0.9, count=3)  # 6%
    v = verdict(rows)
    assert (v.state, v.rule) == ("INCONCLUSIVE", 2)
    assert "cubic:T3_band" in v.tasks


def test_nan_share_counts_only_best_rate():
    rows = make_rows(n_seeds=50)
    # Make lr=1e-2 the best rate for perceptron on T2, then put NaN only at 1e-1.
    set_cell(rows, "perceptron", "T2_xor", lr=1e-2, n_success=40)
    set_cell(rows, "perceptron", "T2_xor", lr=1e-1, nan_flag=True, count=1)
    assert verdict(rows).state == "NO-GO"
    set_cell(rows, "perceptron", "T2_xor", lr=1e-2, nan_flag=True, count=1)  # 2% at best
    v = verdict(rows)
    assert (v.state, v.rule) == ("INCONCLUSIVE", 2)


def test_rule2_is_per_task_not_pooled():
    rows = make_rows(n_seeds=50)
    set_cell(rows, "eml", "T4_stripes", clip_step_frac=0.9, count=6)  # 12% on T4, 4% pooled
    assert verdict(rows).state == "INCONCLUSIVE"


def test_best_rate_chosen_on_training_accuracy_not_success():
    rows = make_rows(n_seeds=10, default=0.0)
    for r in rows:
        if r.unit == "eml" and r.task == "T2_xor":
            if r.lr == 1e-1:  # high train acc, but test fails -> no success
                r.train_acc, r.test_acc, r.success = 1.0, 0.5, False
            elif r.lr == 1e-2 and r.seed < 5:
                r.train_acc, r.test_acc, r.success = 1.0, 1.0, True
    assert best_rates(rows)[("eml", "T2_xor")] == 1e-1
    assert verdict(rows).state == "NO-GO"


def test_tie_goes_to_smaller_rate():
    rows = make_rows(n_seeds=10)
    assert best_rates(rows)[("eml", "T2_xor")] == 1e-3


def test_report_contains_scope_sentence():
    rows = make_rows(n_seeds=10)
    text = render_report(rows, verdict(rows), "| g1 |")
    assert SCOPE_SENTENCE in text


def test_two_family_verdict_and_controls():
    from eml_study.family import TWO

    rows = []
    for u in TWO.units:
        for t in TWO.tasks:
            for lr in LRS:
                k = 0 if t in TWO.controls else 10
                rows += [_rec(u, t, lr, s, s < k) for s in range(20)]
    assert verdict(rows, TWO).state == "NO-GO"
    set_cell(rows, "eml", "U5_wave", n_success=15)  # 75% vs 50%
    v = verdict(rows, TWO)
    assert (v.state, v.tasks) == ("GO", ["U5_wave"])
    set_cell(rows, "ellipse", "C1_ring", lr=1e-2, n_success=1)
    assert verdict(rows, TWO).state == "INVALID"
