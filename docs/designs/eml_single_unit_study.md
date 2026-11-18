# EML Unit vs. Perceptron: Single-Unit Capability Study

*   **Author:** darhnoel
*   **Status:** Approved
*   **Date:** 2026-11-17

--------------------------------------------------------------------------------

## Overview

This study asks two narrow questions. First, what can a single unit built on
the EML (Exp-Minus-Log) operator, `eml(x, y) = e^x − ln(y)`, classify that a
single perceptron cannot? Second, is any of that ability unique to EML, or do
well-known single-unit activations give the same result?

The EML operator comes from Odrzywołek, "All elementary functions from a single
operator" (arXiv:2603.21852, 2026). The paper shows that trees of EML nodes,
together with the constant 1, can express the 36 scientific-calculator
primitives listed in its Table 1. It does not test EML as a classification
unit. This study tests whether one EML unit offers any single-unit advantage
over well-known activations. It does not decide whether multi-layer EML
networks are useful.

The output is a verdict computed by code (GO, NO-GO, INCONCLUSIVE or INVALID,
see G5), backed by:

*   a mathematical description of the decision boundaries one EML unit can draw;
*   measured results on a fixed set of small 2-D classification tasks;
*   the same measurements for a perceptron and for five other activations with
    the same number of parameters. Three of them (Gaussian, quadratic and |x|)
    already solve XOR with one unit. The other two, a cubic and a sine, draw
    three or more parallel lines.

## The Problem

A perceptron computes `step(w·u + b)`. Its decision boundary is always one
straight line (one hyperplane in higher dimensions). Because of this it cannot
solve XOR or any other task that is not linearly separable. Minsky and Papert
made this limit well known in 1969.

The EML paper motivates a candidate replacement unit. Its claim is about what
EML can express (exact elementary functions), not about how well it
classifies. Three facts from the paper make "replace the perceptron" an open
question rather than a result:

1.  **The paper has no classification experiments.** It does not compare EML
    with perceptrons or MLPs. Its training experiments are symbolic regression
    only. It argues that any standard network is a special case of an EML tree,
    but it does not test this.
2.  **EML has no cheap weighted sum.** A perceptron is built around `w·u + b`.
    In pure EML, `x + y` takes 9 eml nodes (K = 19, the shortest form found by
    the paper's direct search). So an EML unit that accepts weighted inputs has
    to add affine maps from outside the operator, and the design of those maps
    changes what the unit can do.
3.  **Training from random starts gets much harder with depth.** The paper's
    gradient-trained trees recover the exact target from random initialization
    100% of the time at depth 2, about 25% at depths 3–4, under 1% at depth 5,
    and 0 of 448 attempts at depth 6. From perturbed correct weights they
    converge at every depth tested. The paper's training code uses complex
    arithmetic, clamping to prevent overflow, and NaN checks.

Getting past XOR alone does not show that EML is useful. Single units with a
Gaussian (`e^(−s²)`), quadratic (`s²`) or absolute-value (`|s|`) activation
already solve XOR, because a U-shaped or bump-shaped function of one line
accepts a band between two parallel lines. An activation that crosses a
threshold three times, such as the cubic `s³ − s`, draws three parallel lines.
`sin(s)` draws any number of them. To show a single-unit advantage, EML has to
do the same tasks more reliably than these activations.

A NO-GO means EML has no single-unit advantage over these activations. A GO
names the tasks on which it trained more reliably. Either way, the result says
nothing about EML inside multi-layer networks: a single ReLU unit would also
get NO-GO here.

## Goals

*   **G1: Describe the boundaries analytically.** For each of the seven unit
    types (EML, perceptron, Gaussian, quadratic, |x|, cubic, sine), write down
    the set of decision boundaries `{u : g(w·u + b) = t}` it can draw in 2-D
    input space: how many parallel lines it gives, which ratios of stripe widths
    are possible, and in what order the class regions appear.
*   **G2: Measure separability on a fixed task set.** For each of the seven unit
    types, measure training and test accuracy on six 2-D tasks: linearly
    separable (T1), XOR as four clusters (T2), band (T3), three-stripe (T4),
    concentric circles (T5) and two moons (T6). Every unit gets the same data,
    the same parameter count, the same optimizer and the same training budget.
*   **G3: Measure how reliably each unit trains.** For each (unit, task) pair,
    report the fraction of random starting points that succeed within the
    budget. Each pair runs 100 seeds at each of 3 learning rates. G3 and G5 use
    the 100 seeds at the pair's best learning rate.
*   **G4: Measure numerical stability.** For each unit, count training runs that
    produced overflow, NaN or infinite values and runs that were clipped, and
    report how often any clamp or guard changed the computed output. For EML,
    also record how close each run came to the pole (`min_abs_s`,
    `max_abs_gprime`).
*   **G5: Give a verdict against criteria fixed in this version of the
    document.**
    *   **Definitions:**
        *   A run **succeeds** if it reaches ≥ 99% training accuracy and ≥ 95%
            test accuracy without a NaN. The test condition reduces the chance
            that a unit counts as a success by fitting single points inside
            very thin stripes.
        *   A unit's **best learning rate** for a task is the rate with the
            highest fraction of seeds reaching ≥ 99% training accuracy, with
            ties going to the smaller rate. Its **success rate** on that task
            is the fraction of its 100 seeds at that rate that succeed. Its
            **best-rate runs** are those 100 runs.
        *   The **evaluated tasks** are T2, T3 and T4, the tasks a perceptron
            cannot solve. T1 is a sanity check: it is reported but does not
            enter the rules. The controls T5 and T6 never count toward GO.
    *   **Rules, applied in this order:**
        1.  **INVALID** if any run on T5 or T6 succeeds, for any unit and at
            any learning rate. No unit can draw those boundaries (G1), so such
            a success points to a bug or a numerical artifact. The run is
            checked and repeated, and no GO or NO-GO is issued.
        2.  **INCONCLUSIVE** if, for any unit and any single evaluated task,
            more than 5% of that unit's 100 best-rate runs on that task are
            clipped on more than 50% of their steps, or more than 1% of them
            end with a non-finite value ("NaN" in this document means any
            non-finite value, including ±∞). In either case the
            numerical guards, not the activation, may be deciding the outcome,
            whether the affected unit is EML or a comparison unit.
        3.  **GO** if, on at least one evaluated task (T2, T3 or T4), EML's
            success rate is at least 20 percentage points higher than the best
            comparison unit's. With 100 seeds, a 20-point gap is about 2.8
            standard errors, so across three tasks the chance of a GO from
            noise alone is under 1%. One task is enough so that a lead on T4,
            the only task where EML's three-line shape is needed, counts on
            its own.
        4.  **NO-GO** otherwise.
    *   **Why there is no "only EML can do it" rule:** G1 shows that the cubic
        and the sine can draw every boundary EML can. Among single-sum units
        no task exists that only EML can draw, so such a rule could only fire
        when a comparison unit failed to train. Rule 3 already measures
        training reliability directly.

## Non-Goals

*   **Multi-layer or stacked EML networks.** This study looks at one unit only.
    It makes no claim about deep EML networks.
*   **The two-input EML unit `eml(a·u + b, c·u + d)`.** Deferred to a follow-up
    study (see Alternative C and Future Improvements).
*   **Symbolic regression and exact-formula recovery.** These are the paper's
    focus and would be a separate project that compares EML trees with MLPs on
    recovering formulas.
*   **Real-world datasets or large benchmarks** (MNIST, tabular data, and so on).
    Tasks are small and synthetic so that every boundary can be plotted and
    checked.
*   **Speed, memory or hardware efficiency.** Only which boundaries a unit can
    draw and how reliably it trains are measured.
*   **Proving new results about EML's universality.** The paper's expressivity
    claims are taken as given and are not re-checked here.
*   **Production code or a reusable library.** The output is a research report
    plus the scripts needed to reproduce it.

## Open Questions

None.

## Decisions Recorded

*   **Study scope: one unit, not a network or symbolic regression.** See
    [Study scope](#study-scope) (Alternatives A, B).
*   **Unit form: one shared weighted sum.** The EML unit computes
    `eml(s, s)` with `s = w·u + b`. All seven units have the same parameters
    (`w₁, w₂, b, t, k`), so no unit wins by having more parameters. The
    two-input form is deferred to a follow-up study. See
    [Unit form](#unit-form) (Alternatives C, D).
*   **Values where `s ≤ 0`: real part of complex EML.** The unit uses
    `Re eml(s, s) = e^s − ln|s|`. This is the real part of the paper's
    principal-branch definition. Discarding the imaginary part is a choice made
    by this study. Unlike softplus (Alternative E) or a `+∞` mask
    (Alternative F), it adds a third crossing point for s < 0. See
    [Handling s ≤ 0](#handling-s--0).
*   **Comparison set includes a cubic and a sine.** `s³ − s` and `sin(s)` were
    added so that EML's third line is tested against activations that also draw
    three or more lines, not only against activations that draw two. See
    [Comparison set](#comparison-set) (Alternative G).
*   **Loss: logistic with learned threshold and scale.** All units train on
    binary cross-entropy of `k·(t − g(s))`, with `t` and `k` learned. See
    [Loss and threshold](#loss-and-threshold) (Alternatives H, I).

## Critical User Journeys (CUJs)

### CUJ 1: Reproduce the verdict

*   **Given:** a copy of the project on a machine with Python 3.12 and `uv`.
*   **When:** the researcher runs `uv sync`, then the `boundary_analysis`,
    `train` and `report` modules in that order.
*   **Then:** `results/report.md` states the verdict (GO, NO-GO, INCONCLUSIVE
    or INVALID) with the tasks or rule that triggered it. A second run on the
    same machine with the same library versions gives the same
    `results/results.csv`.

### CUJ 2: Audit a suspicious result

*   **Given:** a finished full run whose verdict is INVALID or INCONCLUSIVE, or
    whose report shows the pole warning (see [Data flow](#data-flow)).
*   **When:** a reviewer opens the flagged plots and the matching rows in
    `results/results.csv`.
*   **Then:** the reviewer has the data to judge whether each flagged run is a
    pole or clipping artifact. That data is the best-seed and worst-seed plots
    for each (unit, task), any other seed's boundary redrawn from its weights
    with `report.py --plot-seed`, the per-run guard counts and clip fraction,
    and, for EML runs, `min_abs_s` and `max_abs_gprime`.

--------------------------------------------------------------------------------

## Technical Plan (Architecture)

### What every unit computes

Every unit has the same structure. Only the activation `g` changes:

```text
input u = (u₁, u₂)
s     = w₁·u₁ + w₂·u₂ + b            weighted sum, same as a perceptron
z     = k · (t − g(s))               logit; t = threshold, k = scale/sign
class = A if z > 0, else B
loss  = binary cross-entropy on z
parameters: w₁, w₂, b, t, k          (5 per unit, all units)
```

| Unit | `g(s)` | Most parallel lines | Region pattern along `s` |
|---|---|---|---|
| Perceptron | `s` | 1 | A \| B |
| Gaussian | `−e^(−s²)` | 2 | B \| A \| B (or reversed) |
| Quadratic | `s²` | 2 | B \| A \| B (or reversed) |
| Absolute value | `\|s\|` | 2 | B \| A \| B (or reversed) |
| Cubic | `s³ − s` | 3 | A \| B \| A \| B (or reversed) |
| Sine | `sin(s)` | unlimited | A \| B \| A \| B \| … (or reversed) |
| EML | `e^s − ln\|s\|` | 3 | A \| B \| A \| B (or reversed) |

The Gaussian is written `−e^(−s²)` here and `e^(−s²)` in the Problem section.
The sign makes no difference, because the learned `k` can take either sign.
The perceptron uses `g(s) = s` with a logistic loss instead of the hard step,
so that every unit trains with the same gradient method. Its `t` duplicates
`b`. It is kept so that every unit has exactly the same parameters.

### Why the EML unit is computed with real numbers

The unit uses the real part of the paper's complex principal-branch value.
Because `s` is always real, this real part is exactly `e^s − ln|s|`. The
imaginary part is `0` for `s > 0` and `−π` for `s < 0`, and it is thrown away.
The unit is therefore computed in ordinary 64-bit floating point. No complex
arithmetic is needed. Discarding the imaginary part is this study's choice,
and the third crossing point comes from it (see
[Risks & Mitigations](#risks--mitigations)).

### What the analysis already predicts (G1)

`boundary_analysis.py` computes the following. The figures below come from an
earlier bisection check and are reproduced by that module.

*   On the positive side, the EML curve has a single minimum where
    `e^s = 1/s`. That is `s* = W(1) = Ω ≈ 0.567143` (the omega constant), with
    `g_min = Ω + 1/Ω ≈ 2.33037`. On the negative side, `g` rises steadily from
    −∞ (as s → −∞) to +∞ (as s → 0⁻). For any `t > g_min` there are exactly
    three crossing points `s₀ < 0 < s₁ < s₂`.
*   The stripe `(s₀, s₁)` contains the pole at `s = 0` (the "pole stripe"). Both
    sides of the pole are in the same class, so the pole does not add a
    boundary.
*   The ratio of stripe widths `(s₁ − s₀)/(s₂ − s₁)` covers all of (0, ∞) as
    `t` varies. It goes to ∞ as `t` falls toward `g_min`, and is about 5.7 at
    t = 2.34, about 1.0 at t = 2.5, about 0.29 at t = 3, about 0.02 at t = 5,
    and about 10⁻⁴ at t = 10.
*   The cubic `s³ − s` also gives three crossing points for
    `|t| < 2/(3√3) ≈ 0.385`, and its ratio also covers all of (0, ∞).
*   `sin(s)` gives any number of evenly repeating crossings. Over the data's
    range it can draw three lines with any ratio of widths, so its family
    contains EML's.

**Prediction:** EML, the cubic and the sine all draw three parallel lines in
A|B|A|B order with any width ratio. No task separates EML from them by
boundary shape, so the verdict depends on training reliability (G5 rule 3).
One difference could affect reliability: near the pole EML's `g'(s)` grows
without bound, while the cubic's and the sine's stay bounded on the data. On
T4 the pole always lies inside the B stripe, which holds training points, so
some points sit near it. No task inside `[−1, 1]²` forces the extreme
squeeze described above, because the width ratios it would need are below what
the data range allows. `min_abs_s` and `max_abs_gprime`, measured at the final
trained weights, record how close the fit puts a training point to the pole.
The experiments test this prediction; they do not assume it.

### Numerical guards

*   **Pole at `s = 0` (EML only):** compute `ln(max(|s|, ε))` with
    `ε = 10⁻¹²`. Count every point-step where the guard changes the value.
    Because this guard rarely fires for continuous `s`, every EML run also
    records, at its final trained weights, `min_abs_s` (the smallest |s| over
    training points) and `max_abs_gprime` (the largest |g'(s)| over training
    points, using the guarded `|s|` and clamped `s`) (G4).
*   **Overflow in `e^s` (EML only):** clamp `s ≤ 350` inside `e^s`. 64-bit
    `exp` overflows just above 709. The per-seed gradient-norm computation
    squares gradient entries that scale with `|k|·e^s`, and those squares
    overflow near `e^{2s}` ≈ 10³⁰⁸ (`s ≈ 354` for `|k| ≈ 1`). The clamp at 350
    leaves headroom. Adam itself only ever sees clipped gradients. Count every
    clamp (G4).
*   **Gradient size (all units):** clip each seed's gradient norm at 10,
    computed over that seed's own 5 parameters. It is never a global norm
    across seeds. Record the fraction of steps on which each run is clipped. G5
    rule 2 checks this fraction for every unit.
*   **NaN or infinity (all units):** before each optimizer step, check each
    seed's loss and gradient. A seed with any non-finite value has its
    parameter *update* masked from then on, so Adam's stored momentum cannot
    move it. It is marked failed and counted (G4). Other seeds keep training.

### Tasks

All tasks are 2-D. Inputs are in `[−1, 1]²`, with 400 training points and 400
test points per task. The data are generated once from a fixed seed. Labels are
noise-free. For T1, T3, T4 and T5, points closer than 0.02 to a true boundary
are dropped. T2's clusters and T6's arcs are separated by at least 0.3 by
construction, so no margin rule is needed for them. A unit whose family
contains the true boundary can reach 100%.

| Task | Rule for class A | Which units can draw it (from G1) |
|---|---|---|
| T1 Linear | `u₁ + 0.5·u₂ > 0.1` | all seven |
| T2 XOR | four clusters (radius 0.15) at `(±0.5, ±0.5)`; A = same-sign corners | all except the perceptron |
| T3 Band | `\|u₁ − u₂\| < 0.4` | all except the perceptron |
| T4 Three-stripe | projection `p = (u₁ + u₂)/√2`; boundaries at `p = −0.6, 0.0, 0.9`, A\|B\|A\|B from low to high `p` | cubic, sine, EML |
| T5 Circles | `‖u‖ < 0.6` | none (control) |
| T6 Two moons | noise-free two moons: upper arc `(cos θ, sin θ)`, lower arc `(1 − cos θ, 0.5 − sin θ)`, θ ∈ [0, π], then `u = 0.6·(x − 0.5, y − 0.25)`; A = upper arc | none at 99% (control) |

T5 and T6 are controls. A unit that succeeds on them is expected to have done
so through a bug or a numerical artifact, for example fitting single points
into very thin stripes next to the EML pole. G5 rule 1 then makes the verdict
INVALID.

### Training procedure

*   **Optimizer:** Adam, full batch, 3000 steps.
*   **Learning rates:** each (unit, task) pair is run at `10⁻³`, `10⁻²` and
    `10⁻¹`. Results are reported at every rate. G5 uses each unit's best rate
    per task, chosen on training accuracy only (see G5 definitions), so test
    data does not influence which rate is used.
*   **Seeds:** 100 seeds per (unit, task, rate). The seed changes only the
    starting weights, never the data.
*   **Starting values:** `w ~ N(0, I)`, `b ~ N(0, 1)`, `k = 1`, and `t` set to
    the median of `g(s)` over the training points at the start. This gives
    every unit a starting split near 50/50, whatever the range of its `g`.
*   **Total size:** 7 units × 6 tasks × 3 rates × 100 seeds = 12,600 runs. The
    100 seeds for one (unit, task, rate) are trained together as a single
    batched tensor, so there are 126 training jobs. This is expected to take
    minutes on a laptop CPU.

### Data flow

```text
 boundary_analysis.py ──► results/boundaries.csv   (crossings, width ratios)
                      └─► results/g1_summary.md    (G1 tables, all seven units)

 tasks.py ──(fixed seed)──┐
 units.py ──(7 g(s), guards)──┤
                          ▼
 train.py ── for each (unit, task, rate): 100 seeds in one batch
                          │
                          └─► results/results.csv  (one row per run: accuracies,
                                success, NaN flag, guard and clip counts,
                                EML pole diagnostics, final w1 w2 b t k)

 report.py ── reads results/results.csv, results/boundaries.csv,
              results/g1_summary.md
           ├─► results/report.md   verdict (G5 rules) with the scope
           │                       sentence, success-rate tables (G2, G3),
           │                       stability tables (G4), G1 tables, warning
           └─► results/plots/      best and worst seed per (unit, task);
                                   any seed on request with --plot-seed
```

`boundary_analysis.py` does not depend on training, so it runs first.
`report.py` computes the verdict from `results/results.csv` using the G5 rules
exactly as written. Nobody chooses it after looking at the plots. The verdict
line in `report.md` always carries the scope sentence: "This verdict concerns
one real-valued, shared-sum EML unit. It does not decide two-input or
multi-layer EML."

The report shows one warning, at the top of `report.md`, which never changes
the verdict: any best-rate EML success whose final `max_abs_gprime` is above
10⁶, which can point to a pole-driven solution. `report.py` renders a plot for
each warned run, and the warning links it.

### Tools

Python 3.12 with PyTorch (64-bit floats, batched over seeds) for training,
NumPy for data generation, and Matplotlib for plots. PyTorch is used because
its automatic gradients and batching avoid hand-written derivatives for seven
activations. No code from the paper is needed.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Batched seeds affect each other, so success rates are not per-seed results | Per-seed loss summing, per-seed gradient clipping, per-seed update masking and per-seed random generators; a test checks that a seed trained in a batch matches the same seed trained alone |
| The verdict rule has gaps that let an artifact produce GO | Four-state G5 applied in a fixed order; controls excluded from GO and made to force INVALID; best-rate scope written into G5 |
| A result is credited to EML when it comes from this study's `Re` / `ln\|s\|` construction | The study states that it tests the real function `e^s − ln\|s\|`, not EML inside complex-valued trees; the verdict line carries a scope sentence |
| Single-unit results are read as a verdict on EML networks | NO-GO is limited in scope to "no single-unit advantage" (Overview, Problem, verdict line) |
| The comparison set leaves out an activation that matches EML | The cubic and the sine, which both contain EML's boundary family, are included |
| A pole artifact counts as a success | Test-accuracy bar of 95%; INVALID on any control success; `min_abs_s`, `max_abs_gprime` and the pole warning for EML |
| Numerical guards decide the outcome, for EML or for a comparison unit | G5 rule 2 (INCONCLUSIVE) checks heavy clipping and NaNs for every unit; `exp` clamp at 350 so squared gradients stay finite |
| `t` and `k` work on different scales for different activations, so one Adam step changes some units' boundaries more than others', and a reliability gap (rule 3) partly reflects that | `t` starts at the median of `g(s)`; three learning rates per unit with the best chosen per unit; results are reported at every rate so the effect is visible. The remaining bias is accepted, and `report.md` states it in a fixed limitations paragraph |
| Results differ across machines or library versions | The reproducibility claim covers the same machine and the same library versions only |
| Task data is underspecified | Every task rule, including the T6 generator and its scaling, is stated in [Tasks](#tasks) |

## Alternatives Considered

The alternatives below were weighed during the drafting discussion on
2026-11-17. No separate brainstorm document exists, so this section is the
record of what was rejected and why.

### Study scope

#### Alternative A: EML neuron in layered networks

Each unit computes `eml(a·u + b, softplus(c·u + d))` with real-valued weights,
and units are stacked like an MLP and compared with an MLP of the same size.

*   **Pros:** Closest to a literal "drop-in replacement for the perceptron".
    Results are easy to measure (accuracy, parameter count, training
    stability).
*   **Cons:** It competes with ReLU and GELU as an unusual two-input
    activation. An MLP with any non-polynomial activation can already
    approximate any continuous function (the universal approximation
    theorem), so EML's expressivity result gives it no advantage here. The
    unbounded `e^x` forces clipping or normalization, and that changes the
    function being tested. (Real-valued input weights also give up the paper's
    exact-formula recovery. This applies to the chosen single unit too, and the
    single-unit study accepts that.)
*   **Reason for Rejection:** Rejected by author: a single-unit study is
    cheaper and answers a narrower question first. Its result does not decide
    whether a layered network is worth building.

#### Alternative B: EML tree against MLP for symbolic regression

Use the paper's trees, where each input of a node is a softmax-weighted mix of
{1, x, child output} that is rounded to a single choice after training.
Compare them with an MLP on recovering and extrapolating elementary laws.

*   **Pros:** Tests what the paper actually claims EML is good at (exact
    recovery and extrapolation). The paper's own results give a starting
    point.
*   **Cons:** It is a symbolic-regression project competing with tools such as
    PySR and AI Feynman, not a perceptron replacement. The paper reports
    recovery from random initialization of about 25% at depth 4, under 1% at
    depth 5 and 0 of 448 at depth 6. So any law deeper than about 4 levels is
    out of reach from random initialization. It needs complex arithmetic.
*   **Reason for Rejection:** Rejected by author: it does not answer the
    perceptron question, and its main limit (depth) is already measured in the
    paper.

### Unit form

#### Alternative C: two independent weighted sums, `eml(a·u + b, c·u + d)`

*   **Pros:** With `z₁ = a·u + b` and `z₂ = c·u + d`, the boundary is the curve
    `z₁ = ln(t + ln z₂)`, which none of the comparison units draw. It contains
    the shared-sum unit as the special case `a = c, b = d`.
*   **Cons:** Twice the input parameters of a perceptron (6 against 3 in 2-D),
    so every comparison unit would also need a second weighted sum to keep the
    comparison fair. The curved boundary could also come close to the moons
    control, so the controls would need redesigning.
*   **Reason for Rejection:** Deferred by author to a follow-up study. The
    shared sum keeps all units at the same 5 parameters with no matching scheme
    and keeps the controls valid. This leaves curved boundaries untested (see
    Future Improvements).

#### Alternative D: paper-style depth-2 tree as one unit

*   **Pros:** The most faithful to arXiv:2603.21852.
*   **Cons:** Its input weights are limited to a simplex and rounded to
    {0, 1}, so it cannot apply arbitrary affine maps to the input. It could fail
    the linearly separable task that a perceptron passes, so the comparison
    would mostly measure the restricted weights, not the operator.
*   **Reason for Rejection:** Rejected by author: without arbitrary input
    weights it cannot shift or rotate a line, so it is not a perceptron
    analog.

### Handling `s ≤ 0`

#### Alternative E: softplus inside the log, `e^s − ln(softplus(s))`

*   **Pros:** Real, smooth and defined everywhere, with gradients everywhere.
*   **Cons:** It changes the function wherever softplus(s) differs noticeably
    from s (for s below about 2). Its boundary is a band (two lines), the same
    as Gaussian, quadratic and |x|.
*   **Reason for Rejection:** Rejected by author: a unit that draws only a band
    is just another band activation, so the study would compare EML with only
    the units that match its shape. The real-part unit can draw three lines,
    though the same caveat about attribution applies to it (see
    [Risks & Mitigations](#risks--mitigations)).

#### Alternative F: exact EML on `s > 0`, `+∞` for `s ≤ 0`

*   **Pros:** Keeps the operator unchanged where it is real-valued. At `s = 0`
    it matches the paper's `ln 0 = −∞` convention. For `s < 0` it departs from
    the paper, which uses the complex principal branch there.
*   **Cons:** The boundary is a band, the same as existing activations. There
    is no gradient for `s ≤ 0`, so points stuck there cannot be moved, and G3
    success rates would probably fall for reasons unrelated to EML's shape.
*   **Reason for Rejection:** Rejected by author: least likely to show
    anything, and the dead region would make reliability results hard to
    interpret.

### Comparison set

#### Alternative G: keep only perceptron, Gaussian, quadratic and |x|

*   **Pros:** A smaller study. These are the standard single-unit XOR
    solvers.
*   **Cons:** None of them can draw three parallel lines. On the three-stripe
    task EML would appear to beat every comparison unit, because the set leaves
    out any activation that crosses a threshold three times, not because EML is
    special.
*   **Reason for Rejection:** Rejected by author: a GO would mostly reflect the
    choice of comparison units. The cubic `s³ − s` and the sine `sin(s)` were
    added.

### Loss and threshold

#### Alternative H: hinge loss, learned threshold

*   **Pros:** Closest to the classic perceptron rule, because only
    misclassified points and points inside the margin push the weights.
*   **Cons:** Hinge loss stops pushing a point once it is past margin 1, while
    logistic loss keeps a small push on every point. Near the pole both losses
    are scaled by `g'(s)`, which grows without bound as s → 0, so neither
    avoids large gradients there. Whether logistic loss's continued push keeps
    EML out of very thin stripes is a hypothesis, not a result.
*   **Reason for Rejection:** Rejected by author: no advantage over logistic
    loss for this study, and it drops the push on points already past the
    margin.

#### Alternative I: logistic loss, fixed threshold per unit

*   **Pros:** Fewer free parameters. It tests the geometry of the activation
    more directly.
*   **Cons:** With `t` fixed, the three crossing points are fixed numbers. The
    weights can only scale and shift the whole pattern, so the ratio of the two
    stripe widths is locked. EML and the cubic could then fail the three-stripe
    task because of the chosen `t`, not because of the activation.
*   **Reason for Rejection:** Rejected by author: the locked width ratio would
    confound the result.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Reproducibility:** data seeds, per-seed generators and float64 are fixed,
    so reruns give identical `results/results.csv` on the same machine and
    library versions.
*   **Runtime:** 126 batched training jobs of 3000 full-batch steps on 400
    points. Expected to take minutes on a laptop CPU, with no GPU needed.
*   **Security and privacy:** not applicable. All data is synthetic.

## Additional Resources

*   Odrzywołek, A. "All elementary functions from a single operator."
    arXiv:2603.21852, 2026. <https://arxiv.org/abs/2603.21852>
*   Minsky, M. and Papert, S. *Perceptrons*, 1969 (the single-perceptron XOR
    limit).

## Future Improvements

*   **Two-input EML unit (Alternative C).** It is the only single-unit form
    considered here that draws curved boundaries, and the form closest to how
    the paper uses EML. A follow-up study needs comparison units with a matched
    second weighted sum and controls that a curved boundary cannot fit.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Environment

Requires Python 3.12 and `uv`. `uv sync`
installs them into a project virtual environment (`.venv/`) from
`pyproject.toml` and records exact versions in `uv.lock`. Every command below
runs through `uv run` from the project root. All paths are relative to the
project root.

### Layout

```text
./
├── pyproject.toml
├── .python-version                            (contains "3.12")
├── uv.lock                                    (generated by uv sync; kept)
├── .venv/                                     (generated by uv sync)
├── README.md
├── docs/designs/eml_single_unit_study.md      (this design)
├── eml_study/
│   ├── __init__.py
│   ├── units.py
│   ├── tasks.py
│   ├── train.py
│   ├── boundary_analysis.py
│   └── report.py
├── tests/
│   ├── test_units.py
│   ├── test_tasks.py
│   ├── test_train.py
│   ├── test_boundary_analysis.py
│   └── test_report.py
└── results/                                   (generated, not hand-edited)
    ├── results.csv
    ├── boundaries.csv
    ├── g1_summary.md
    ├── plots/*.png
    ├── report.md
    └── quick/                                 (smoke-run outputs, same names)
```

The package sits at the project root and is not installed. `python -m` finds it
because it runs from the project root. pytest finds it through
`pythonpath = ["."]` in `pyproject.toml`.

### Phases

Each phase ends with `uv run pytest` passing before the next one starts.

1.  `pyproject.toml`, `.python-version`, `eml_study/__init__.py`,
    `eml_study/units.py`, `tests/test_units.py`.
2.  `eml_study/tasks.py`, `tests/test_tasks.py`.
3.  `eml_study/boundary_analysis.py`, `tests/test_boundary_analysis.py`.
4.  `eml_study/train.py`, `tests/test_train.py`, then
    `uv run python -m eml_study.train --quick --out-dir results/quick`.
5.  `eml_study/report.py`, `tests/test_report.py`, `README.md`, the
    smoke run, the reproducibility check, then the full run.

### Target Files to Create

#### 1. `./pyproject.toml` (Create)

*   **Type / Scope:** Dependency and test configuration.
*   **Proposed Changes:** `[project]` with `name = "eml-study"`,
    `version = "0.1.0"`, `requires-python = ">=3.12,<3.13"`, and dependencies
    `torch`, `numpy` and `matplotlib` (exact versions are fixed by `uv.lock`).
    `[dependency-groups] dev = ["pytest"]`. No `[build-system]` table, so uv
    does not try to install the project. `[tool.pytest.ini_options]` with
    `testpaths = ["tests"]` and `pythonpath = ["."]`.
*   **Why:** One file lists what the study needs, and `uv.lock` fixes the
    versions behind the "same library versions" promise in CUJ 1.
    `pythonpath` lets the tests import the package without installing it.

#### 2. `./.python-version` (Create)

*   **Type / Scope:** Interpreter pin for uv.
*   **Proposed Changes:** the single line `3.12`.
*   **Why:** It stops uv from choosing a newer interpreter.

#### 3. `./README.md` (Create)

*   **Type / Scope:** Documentation.
*   **Proposed Changes:** One paragraph on the question being studied, a link
    to `./docs/designs/eml_single_unit_study.md`, and the four commands in run
    order: `uv sync`, `uv run python -m eml_study.boundary_analysis`,
    `uv run python -m eml_study.train`, `uv run python -m eml_study.report`.
    State that `results/report.md` holds the verdict.
*   **Why:** Anyone rerunning the study needs the order of steps.

#### 4. `./eml_study/__init__.py` (Create)

*   **Type / Scope:** Package marker.
*   **Proposed Changes:** a module docstring and
    `torch.set_default_dtype(torch.float64)`.
*   **Why:** Tensors created without an explicit dtype then default to
    float64, which the `exp` clamp at 350 and `ε = 10⁻¹²` assume.

#### 5. `./eml_study/units.py` (Create)

*   **Type / Scope:** Python module, the seven activations and the shared unit.
*   **Proposed Changes:**
    *   `UNITS: dict[str, Activation]` with keys `perceptron`, `gaussian`,
        `quadratic`, `abs`, `cubic`, `sine`, `eml`. `Activation` holds `name`,
        `g(s) -> Tensor` and `gprime(s) -> Tensor`. Both accept a tensor of any
        shape and return the same shape.
    *   `g` definitions exactly as in the Technical Plan table. For `eml`:
        `exp(clamp(s, max=350)) − log(clamp(|s|, min=1e-12))`, with
        `gprime = exp(clamp(s, max=350)) − 1/(sign(s)·clamp(|s|, min=1e-12))`
        (sign(0) taken as +1).
    *   `guard_counts(s) -> (pole_guard_hits, exp_clamp_hits)` per seed for
        `eml`: the number of entries in each seed's row with `|s| < 1e-12`,
        and with `s > 350`. Other units return zeros.
    *   `pre_activation(params, X) -> s`, where `params` has shape
        `(n_seeds, 5)` ordered `[w1, w2, b, t, k]` and `X` has shape
        `(n_points, 2)`. It computes `s` element by element,
        `X[:,0]·w1 + X[:,1]·w2 + b` (no matrix multiply), giving shape
        `(n_seeds, n_points)`. The arithmetic for one seed is then the same
        whether it runs alone or in a batch.
    *   `logits(params, X, activation) -> z = k·(t − g(s))`, shape
        `(n_seeds, n_points)`.
    *   `N_PARAMS = 5`.
*   **Why:** One shared `logits` function means every unit differs only in
    `g`. That is the fairness condition the comparison relies on.

#### 6. `./eml_study/tasks.py` (Create)

*   **Type / Scope:** Python module, synthetic data.
*   **Proposed Changes:**
    *   `TASKS = ["T1_linear", "T2_xor", "T3_band", "T4_stripes",
        "T5_circles", "T6_moons"]`, `CONTROL_TASKS = {"T5_circles",
        "T6_moons"}`, `EVALUATED_TASKS = {"T2_xor", "T3_band",
        "T4_stripes"}`, `SANITY_TASKS = {"T1_linear"}`.
    *   `make_task(name, split, n=400, data_seed=0) -> (X, y)`, where `X` is a
        NumPy float64 array of shape `(n, 2)` and `y` a float64 array of 0s
        and 1s (1 = class A). `split == "train"` uses
        `np.random.default_rng(data_seed)`, and `split == "test"` uses
        `np.random.default_rng(data_seed + 1)`.
    *   T1, T3, T4 and T5: draw batches of `4n` points uniformly in
        `[−1, 1]²`, drop points within 0.02 of a true boundary, and repeat
        until at least `n` remain. Keep the first `n`.
    *   T2: `n/4` points per cluster, area-uniform in a disc of radius 0.15
        (radius `0.15·√U`, angle `2πV`, with U and V uniform on [0, 1]).
    *   T6: `n/2` points per arc, θ uniform on [0, π], with the generator and
        scaling in the Technical Plan task table.
    *   Labels exactly as in the Technical Plan task table.
*   **Why:** Fixed, noise-free data with a margin means that a unit whose
    boundary family contains the true boundary can reach 100%. Any failure is
    then attributable to training or to the boundary family, not to label
    noise.

#### 7. `./eml_study/train.py` (Create)

*   **Type / Scope:** Python module and command-line entry point.
*   **Proposed Changes:**
    *   `init_params(activation, X_train, seeds) -> Tensor(n_seeds, 5)`. For
        each seed, `gen = torch.Generator().manual_seed(seed)`, then
        `torch.randn(3, generator=gen)` gives `w1, w2, b` in that order.
        `k = 1`, and `t = torch.quantile(g(s), 0.5)` over that seed's
        training points.
    *   `train_batch(activation, task, lr, seeds, steps=3000, clip_norm=10.0,
        init=None) -> list[RunRecord]`. `task` is a task name, and the
        function calls `make_task` for both splits. `init` is an optional
        `(n_seeds, 5)` tensor that replaces `init_params`.
    *   Each step runs these operations in order:
        1.  Forward pass on the training points.
        2.  Per-seed mean binary cross-entropy with logits. Sum over seeds.
            Backward.
        3.  Any seed whose loss or gradient row is non-finite is marked
            frozen. The mark is permanent.
        4.  Zero the gradient rows of frozen seeds.
        5.  Per-seed clip: for each row, if its L2 norm is above `clip_norm`,
            multiply the row by `clip_norm / norm`. Count clipped steps per
            seed.
        6.  `optimizer.step()`, using `torch.optim.Adam` with default betas
            and eps and no weight decay.
        7.  Copy the saved parameter values back into frozen rows, so Adam's
            stored momentum cannot move them.
        8.  Add `guard_counts` to each unfrozen seed's running totals. Only
            training forward passes count.
    *   After the last step: compute training and test accuracy,
        `final_loss` (the training loss at the final weights), and, for EML,
        `min_abs_s` and `max_abs_gprime` over training points at the final
        weights. If any final `z` is non-finite, set `nan_flag`.
    *   `RunRecord` (a dataclass), one row in `results.csv`: `unit, task, lr,
        seed, train_acc, test_acc, success, final_loss, nan_flag,
        pole_guard_frac, exp_clamp_hits, clip_step_frac, min_abs_s,
        max_abs_gprime, w1, w2, b, t, k`.
        *   `success` is
            `train_acc ≥ 0.99 and test_acc ≥ 0.95 and not nan_flag`.
        *   `pole_guard_frac` is guard hits divided by
            `steps × n_train_points`.
        *   `clip_step_frac` is clipped steps divided by `steps`.
        *   `min_abs_s` and `max_abs_gprime` are empty for units other than
            EML.
        *   Frozen runs keep the accuracy of their frozen weights. They count
            toward best-rate selection like any other run.
    *   `write_csv(rows, path)` and `read_csv(path) -> list[RunRecord]`, using
        the standard library `csv` module.
    *   Command line: `python -m eml_study.train [--quick] [--out-dir DIR]`.
        The default is the full grid (7 units × 6 tasks × rates
        `{1e-3, 1e-2, 1e-1}` × seeds 0–99 × 3000 steps), written to
        `DIR/results.csv` (default `results/`). `--quick` uses seeds 0–1 and
        200 steps. The script creates `DIR` if it is missing.
*   **Why:** This file carries out G2–G4. Per-seed loss summing, per-seed
    clipping, update masking, per-seed generators and the element-by-element
    computation of `s` keep each seed's result independent of batching.
    `test_train.py` checks this.

#### 8. `./eml_study/boundary_analysis.py` (Create)

*   **Type / Scope:** Python module and command-line entry point (G1).
*   **Proposed Changes:**
    *   `OMEGA = 0.5671432904097838`.
    *   `crossings(unit, t, s_range=(−20.0, 20.0)) -> list[float]`. Uses
        unguarded `g` and bisection to `1e-14`, with these brackets:
        *   **EML:** `[−20, −1e-300]`, `[1e-300, Ω]` and `[Ω, 20]`.
        *   **Cubic:** split at `±1/√3`.
        *   **Sine:** brackets between consecutive extrema,
            `[−π/2 + jπ, π/2 + jπ]`, clipped to `s_range`. A root that falls
            on a bracket end is counted once.
        *   **Perceptron, Gaussian, quadratic, |x|:** closed forms.
    *   `stripe_ratio(roots) -> (s₁ − s₀)/(s₂ − s₁)` for three-root cases.
    *   The `t` grids are:
        *   EML: `{2.34, 2.5, 3, 5, 10, 20}`, all below the guarded `g`'s
            ceiling of about 28.6 at `|s| = 1e-12`.
        *   Cubic: `{−0.38, −0.2, 0, 0.2, 0.38}`.
        *   Sine: `{−0.9, 0, 0.9}`.
        *   Perceptron: `{0}`.
        *   Gaussian: `{−0.5}`.
        *   Quadratic and |x|: `{0.25, 0.5}`.
    *   Command line: `python -m eml_study.boundary_analysis [--out-dir DIR]`
        writes two files:
        *   `DIR/boundaries.csv`, with columns
            `unit,t,n_roots,s0,s1,s2,ratio` (`s0`–`s2` and `ratio` are left
            empty when there are not three roots);
        *   `DIR/g1_summary.md`, with columns
            `unit | most lines | region pattern | width ratios seen`.
*   **Why:** G1 is part of the deliverable. The claims that the cubic covers
    all width ratios and that the sine's family contains EML's must be
    reproducible from code, not only stated in this design.

#### 9. `./eml_study/report.py` (Create)

*   **Type / Scope:** Python module and command-line entry point (G2–G5).
*   **Proposed Changes:**
    *   `best_rates(rows) -> dict[(unit, task), lr]` and
        `success_rates(rows, best) -> dict[(unit, task), float]`, following
        the G5 definitions. Rates are the number of successes divided by the
        number of seeds present, so a quick run with 2 seeds works.
    *   `Verdict` (a dataclass) with fields `state` (`"GO"`, `"NO-GO"`,
        `"INCONCLUSIVE"` or `"INVALID"`), `rule` (1–4) and `tasks` (the tasks
        that triggered it).
    *   `verdict(rows) -> Verdict`, which applies G5 rules 1–4 in order,
        exactly as written. The fixtures in `test_report.py` pin down its
        behavior.
    *   `render_report(rows, verdict, g1_markdown) -> str`, which builds the
        text of `report.md` without plotting. In order, it contains:
        *   the pole warning, if any (see [Data flow](#data-flow));
        *   the verdict line with the scope sentence, and the triggering rule
            or tasks;
        *   success rates per (unit, task) at every rate and at the best rate;
        *   median train and test accuracy per (unit, task, rate);
        *   the stability table: per unit and task, the NaN share, the share
            of runs with `clip_step_frac > 0.5` (the quantity rule 2 checks)
            and the median `clip_step_frac`; for EML also the median
            `pole_guard_frac`, total `exp_clamp_hits`, smallest `min_abs_s`
            and largest `max_abs_gprime`;
        *   the G1 table from `g1_summary.md`;
        *   the fixed limitations paragraph on `t`/`k` scale (see Risks &
            Mitigations);
        *   plot links.
    *   Plots use Matplotlib's `Agg` backend:
        *   for each (unit, task), the decision regions on a 200×200 grid over
            `[−1, 1]²` for the best and worst seed at the best rate, ranked by
            test accuracy, then by `final_loss`, then by lowest seed id, with
            test points drawn on top;
        *   one plot for each run named in the pole warning;
        *   one bar chart of best-rate success rates per task by unit.
    *   Command line: `python -m eml_study.report [--input PATH]
        [--out-dir DIR] [--plot-seed UNIT TASK LR SEED]`. The defaults are
        `results/results.csv` and `results/`. The script reads
        `DIR/g1_summary.md`, writes `DIR/report.md` and `DIR/plots/`, and
        creates the directories if they are missing. `--plot-seed` redraws one
        run from its weights and exits.
*   **Why:** The verdict has to be computed by fixed code from fixed data, as
    stated in [Data flow](#data-flow).

#### 10. `./tests/test_units.py` (Create)

*   **Type / Scope:** Unit tests.
*   **Proposed Changes:**
    *   **EML values:**
        *   At its minimum, `UNITS["eml"].g(0.5671432904) ≈ 2.3303661248`
            (tolerance 1e-9).
        *   `g(−1) ≈ e⁻¹ = 0.3678794412` (tolerance 1e-10).
        *   `g` is increasing on a grid over `(−5, −1e-6)`.
    *   **Guards:**
        *   At `s = 1e-15`, `guard_counts` reports one pole hit and `g` is
            finite.
        *   At `s = 400`, it reports one clamp hit, and `g` and its autograd
            gradient are finite.
    *   **Cubic and sine:** the cubic has roots at `−1, 0, 1` for `t = 0`, and
        `UNITS["sine"].g(π/2) = 1`.
    *   **Shapes and gradients:**
        *   Every unit has `N_PARAMS = 5`.
        *   `logits` returns shape `(n_seeds, n_points)`.
        *   Its gradients are finite for all seven units on random inputs.
    *   **EML `gprime`:** at `s ∈ {−1, 0.5, 2}`, it equals `e^s − 1/s`
        (tolerance 1e-12).
*   **Why:** The G1 predictions and the training results depend on these
    values.

#### 11. `./tests/test_tasks.py` (Create)

*   **Type / Scope:** Unit tests.
*   **Proposed Changes:**
    *   **Shape and determinism:**
        *   Each task returns 400 points with `X` in `[−1, 1]²`.
        *   The same seed gives the same data.
        *   Train and test differ.
    *   **Margins:**
        *   For T1, T3, T4 and T5, no point lies within 0.02 of a true
            boundary.
        *   For T2 and T6, the two classes are at least `0.3 − 1e-9` apart.
    *   **Class balance:** each class has between 20% and 80% of the points.
    *   **T4 pattern:** the class sequence along `p` is A | B | A | B, with
        changes within the 0.02 margin of −0.6, 0.0 and 0.9.
*   **Why:** The claim that a unit whose family contains the true boundary can
    reach 100% depends on these properties.

#### 12. `./tests/test_train.py` (Create)

*   **Type / Scope:** Integration tests, small budgets.
*   **Proposed Changes:**
    *   **Seed independence:** training seeds 0–3 together gives final
        parameters equal (tolerance 1e-10) to training each seed alone. Run
        for the `eml`, `cubic` and `sine` units on T4, 100 steps.
    *   **Smoke:** the perceptron on T1 succeeds in at least 8 of 10 seeds at
        `lr = 1e-1` within 1000 steps.
    *   **NaN masking:**
        *   Setup: the perceptron on T1, seeds 0–2, with `init` equal to the
            normal starting values except that seed 1's `k = inf`.
        *   Seed 1 ends with `nan_flag = True`, and its parameters equal its
            starting row (compared after `nan_to_num`).
        *   Seeds 0 and 2 change.
    *   **Clip counting:** with `clip_norm = 1e-6`, every run has
        `clip_step_frac = 1.0`.
*   **Why:** These tests check that success rates are per-seed results, and
    that the masking and clipping guards behave as G5 rule 2 assumes.

#### 13. `./tests/test_boundary_analysis.py` (Create)

*   **Type / Scope:** Unit tests.
*   **Proposed Changes:**
    *   **EML:**
        *   `t = 2.5` gives a ratio of `0.9989 ± 1e-3`.
        *   `t = 3` gives `0.2875 ± 1e-3`.
        *   `t = 2.0` (below the minimum) gives exactly one crossing, near
            −0.287.
    *   **Other units:**
        *   The cubic at `t = 0` gives a ratio of 1.0.
        *   The perceptron always gives one crossing.
        *   The sine at `t = 0` over `(−20, 20)` gives exactly the 13 multiples
            of π in that range.
*   **Why:** These check two of the ratios quoted in the Technical Plan and the
    root finder's handling of each bracket type.

#### 14. `./tests/test_report.py` (Create)

*   **Type / Scope:** Unit tests on hand-made rows.
*   **Proposed Changes:**
    *   **Fixture builder:** `make_rows(...)` builds a full 7 × 6 × 3 grid with
        10 seeds per cell, and lets a test override the success, accuracy,
        `nan_flag` and `clip_step_frac` of chosen cells.
    *   **Fixture cases:**
        1.  EML 25 points ahead of every comparison unit on T4 only gives GO
            (rule 3, tasks = [T4]).
        2.  EML 15 points ahead on each of T2, T3 and T4, and 40 points ahead
            on T1, gives NO-GO. T1 does not enter the rules.
        3.  One successful run on T5 gives INVALID, even when rule 3 would
            give GO. This also holds when that run is at a learning rate that
            is not the unit's best.
        4.  On T3, 6% of the cubic's best-rate runs clipped on more than 50% of
            steps gives INCONCLUSIVE. This shows the clip check covers
            comparison units.
        5.  On T2, 2% of one unit's best-rate runs with `nan_flag` gives
            INCONCLUSIVE. 2% only at a learning rate that is not that unit's
            best does not.
        6.  Per-task, not pooled: 12% clipped runs for EML on T4 and 0% on T2
            and T3 gives INCONCLUSIVE, although the pooled share is 4%.
        7.  The best rate is chosen on training accuracy, not on success,
            shown with rows where the two rankings differ.
    *   **Other checks:**
        *   Ties in best learning rate go to the smaller rate.
        *   `render_report` output contains the scope sentence.
*   **Why:** The verdict logic is the study's decision rule. A bug there would
    change the verdict, and no other test checks the rules.

### Verification Plan

*   **Environment:** `uv sync` completes and creates `uv.lock`.
    `uv run python -c "import eml_study"` succeeds.
*   **Unit tests:** `uv run pytest` passes at the end of each phase.
*   **Smoke run:**
    1.  `uv run python -m eml_study.boundary_analysis --out-dir results/quick`
    2.  `uv run python -m eml_study.train --quick --out-dir results/quick`
    3.  `uv run python -m eml_study.report --input results/quick/results.csv --out-dir results/quick`

    These produce `results/quick/report.md` with a verdict line, the success
    rate tables, the accuracy tables, the stability table and the G1 table.
    The full run's files are not touched.
*   **Reproducibility check:** run step 2 of the smoke run a second time into
    `results/quick2/`. `cmp results/quick/results.csv results/quick2/results.csv`
    reports no difference.
*   **Full run:** the same three commands with the default `--out-dir`.
    `results/results.csv` has 12,600 rows. The perceptron's T1 success rate at
    its best learning rate is at least 90%. If it is lower, the run is
    treated as broken and investigated before the verdict is read.
    `results/report.md` states the verdict with the rule or tasks that
    triggered it.

