# EML Two-Input Unit: Single-Unit Follow-Up Study

*   **Author:** darhnoel
*   **Status:** Proposed
*   **Date:** 2026-11-18
*   **Follows:** [eml_single_unit_study.md](./eml_single_unit_study.md)
    (verdict: NO-GO for the shared-sum unit, `results/report.md`)

--------------------------------------------------------------------------------

## Overview

The first study tested `eml(s, s)` with one shared weighted sum and found no
single-unit advantage. On the three-stripe task EML reached 13% against the
sine's 98%. That unit discarded EML's two-input structure. This follow-up tests
the form the first study deferred, which is closer to how the paper uses the
operator:

```text
s₁ = a₁·u₁ + a₂·u₂ + b          (input to e^x)
s₂ = c₁·u₁ + c₂·u₂ + d          (input to ln y)
z  = k · (t − (e^{s₁} − ln|s₂|))
```

The question is the same: does one two-input EML unit classify any task more
reliably than other two-input units of the same size? The answer again comes
from fixed rules computed by code.

## What the two-input EML unit can draw

In the coordinates `(s₁, s₂)`, the boundary `e^{s₁} − ln|s₂| = t` is

```text
|s₂| = h(s₁) = exp(e^{s₁} − t)
```

This is a **funnel** around the line `s₂ = 0`. Its half-width is about `e^{−t}`
far to one side and grows double-exponentially to the other: at t = 2, the
half-width is 0.14 at s₁ = −3, 0.37 at s₁ = 0 and 12 at s₁ = 1.5. One class is
inside the funnel and the other is outside, and the learned sign of `k`
chooses which. When `a` and `c` are not parallel, `(s₁, s₂)` is an affine
change of coordinates, so in input space the unit draws any affine image of
this funnel. It cannot draw a closed curve.

`|s₂|` is the real part of the principal-branch `ln`, as in the first study,
and that remains this study's choice.

## Goals

*   **G1:** for each unit, state the curve family `g_a(s₁) − g_b(s₂) = t` that
    it draws, as in the table below.
*   **G2–G4:** the same as in the first study (accuracy, reliability over 100
    seeds × 3 learning rates, numerical stability), on the units and tasks
    below.
*   **G5 verdict:** the same rules and the same order as in the first study:
    1.  **INVALID** if any run on a control task succeeds, at any rate.
    2.  **INCONCLUSIVE** if, for any unit and evaluated task, more than 5% of
        best-rate runs clip on more than 50% of steps, or more than 1% end
        non-finite.
    3.  **GO** if EML's best-rate success rate is at least 20 points above
        the best comparison unit's on at least one evaluated task.
    4.  **NO-GO** otherwise.

    The success definition (≥ 99% train and ≥ 95% test accuracy) and the
    best-rate rule (chosen on training accuracy) are unchanged.

## Non-Goals

*   Multi-layer networks, complex-valued intermediates and symbolic
    regression, as in the first study.
*   Re-running the shared-sum unit. Its result stands.

## Units

Every unit uses the same template, `z = k·(t − (g_a(s₁) − g_b(s₂)))`, with 8
parameters `a₁, a₂, b, c₁, c₂, d, t, k`. Only the pair `(g_a, g_b)` changes.
The pairs are fixed here, before any run.

| Unit | `g_a(s₁)` | `g_b(s₂)` | Boundary in `(s₁, s₂)` | Why it is included |
|---|---|---|---|---|
| linear | `s₁` | `s₂` | straight line | perceptron baseline at 8 parameters |
| parabola | `s₁²` | `s₂` | parabola | generic open curve |
| ellipse | `s₁²` | `−s₂²` | ellipse (closed) | draws discs |
| hyperbola | `s₁²` | `s₂²` | hyperbola, or two crossing lines at t = 0 | draws quadrant XOR |
| sine-wave | `sin s₁` | `s₂` | wavy curve | periodic boundary |
| exp-half | `e^{s₁}` | `s₂` | exponential curve | ablation: EML with `ln` removed |
| log-half | `s₁` | `ln\|s₂\|` | funnel, half-width `e^{s₁−t}` | ablation: EML with `exp` removed |
| **EML** | `e^{s₁}` | `ln\|s₂\|` | funnel, half-width `exp(e^{s₁}−t)` | the unit under test |

The two ablations matter. If EML beats the generic units but not its own
halves, the advantage belongs to `exp` or `ln` alone, not to the operator.
Rule 3 compares EML against all seven, so that case is a NO-GO.

## Tasks

Inputs are in `[−1, 1]²`, with 400 training and 400 test points from fixed
seeds. Labels are noise-free, and uniform-sampled tasks drop points within
0.02 of the true boundary, as before.

| Task | Rule for class A | Role |
|---|---|---|
| U1 Linear | `u₁ + 0.5·u₂ > 0.1` | sanity check, not in the rules |
| U2 XOR | four clusters (radius 0.15) at `(±0.5, ±0.5)`, A = same-sign corners | evaluated |
| U3 Disc | `‖u‖ < 0.6` | evaluated (ellipse can draw it) |
| U4 Parabola | `u₂ > 1.5·u₁² − 0.5` | evaluated |
| U5 Wave | `u₂ > 0.4·sin(3u₁)` | evaluated |
| U6 Wedge | `\|u₂\| < 0.15 + 0.5·(u₁ + 1)` | evaluated (an opening region, neither EML's shape nor the ellipse's) |
| U7 Moons | the noise-free two moons from the first study | evaluated |
| C1 Ring | `0.35 < ‖u‖ < 0.7` | control (needs two nested closed curves) |
| C2 Checkerboard | 3×3 board on `[−1, 1]²`, A = even cells | control (needs four lines) |

No unit in the table can draw two nested closed curves or a 3×3 grid, so any
success on C1 or C2 points to a bug or an artifact. The tasks are generic
shapes and none of them is EML's funnel. A GO therefore has to come from
training reliability on shapes that several units can approximate.

## Risks

*   **The task choice decides the result.** A task shaped like EML's funnel
    would favor EML, and a disc favors the ellipse. The tasks are fixed here
    and none is EML's own family. The per-task table is reported in full, so
    a reader can see which shapes each unit handles.
*   **More numerical risk than the first study.** The double exponential
    `exp(e^{s₁})` means `e^{s₁}` grows fast. The same guards apply: `s₁` is
    clamped at 350 inside `exp`, `|s₂|` at 1e-12 inside `ln`, gradients are
    clipped per seed, and frozen seeds are masked. Rule 2 catches the case
    where guards decide the outcome.
*   **The `t`/`k` scale bias** from the first study still applies, and the
    report keeps the limitations paragraph.
*   **Approximation tasks.** Moons and the wedge are not exact members of most
    families. Units can still reach 99% by approximation. That is fair, but it
    is a different claim from "draws exactly".

## Implementation plan

The first study's pipeline is reused. The changes are:

1.  **`eml_study/units2.py` (create):** `PAIRS` with the eight units above.
    Each pair holds `g_a`, `g_b` and the EML guards (the `exp` clamp and the
    `ln|·|` floor on any unit that uses them). `pre_activation2` computes
    `(s₁, s₂)` element by element. `logits2` computes
    `z = k·(t − (g_a(s₁) − g_b(s₂)))`. `N_PARAMS2 = 8`. Guard counts use the
    same rules, applied to `s₁` (clamp) and `s₂` (floor).
2.  **`eml_study/tasks2.py` (create):** tasks U1–U7 and C1–C2, with the same
    sampling rules and the same `make_task` interface.
3.  **`eml_study/train.py` (modify):** `train_batch` takes a `family`
    argument (`"single"` or `"two"`) that selects the units, tasks, parameter
    count, starting values and logits. Starting values for two-input units:
    per seed, `torch.randn(6)` gives `a₁, a₂, b, c₁, c₂, d`, `k = 1`, and
    `t` is the median of `g` at the start. `RunRecord` gains `c1, c2, d`,
    which stay empty for single-input runs. `--family` is added to the
    command line. The default stays `single`, so the first study can be
    reproduced unchanged.
4.  **`eml_study/report.py` (modify):** takes `--family`, which selects the
    unit list, comparison set, evaluated tasks and controls. Plotting uses
    the matching logits. The pole warning measures `1/|s₂|` (the `ln` side
    only), which fixes the first study's mixed-cause warning.
5.  **Tests (create/modify):** `test_units2.py` checks values and guards for
    each pair, finite gradients, and that EML's boundary half-width at
    `(s₁, t) = (0, 2)` equals `e^{−1}`. `test_tasks2.py` checks shapes,
    determinism, margins and class balance between 15% and 85%.
    `test_train.py` adds a two-input seed-independence check.
    `test_report.py` adds a family-`two` verdict case.
6.  **Run:** `results/two_input/` holds `results.csv` (8 units × 9 tasks × 3
    rates × 100 seeds = 21,600 runs, about 15 minutes), `report.md` and
    `plots/`.

## Verification

`uv run pytest` passes. The quick run (2 seeds) writes
`results/two_input_quick/report.md` with a verdict line. The full run has
21,600 rows. The linear unit's U1 success rate at its best learning rate is at
least 90%, otherwise the run is treated as broken. No control-task success is
expected.
