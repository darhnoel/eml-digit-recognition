# EML single-unit study: report

> **Pole warning** (does not change the verdict): best-rate EML successes with final max |g'(s)| > 1e+06:

> - T1_linear lr=0.1 seed=54: max |g'(s)| = 2e+07, min |s| = 0.0279 ([plot](plots/warn_T1_linear_0.1_54.png))
> - T1_linear lr=0.1 seed=81: max |g'(s)| = 1.65e+07, min |s| = 0.141 ([plot](plots/warn_T1_linear_0.1_81.png))
> - T1_linear lr=0.1 seed=91: max |g'(s)| = 1.68e+07, min |s| = 0.00646 ([plot](plots/warn_T1_linear_0.1_91.png))

## Verdict: **NO-GO** (G5 rule 4)

EML did not lead the best comparison unit by 20 points on any evaluated task.

*This verdict concerns one real-valued, shared-sum EML unit. It does not decide two-input or multi-layer EML.*

Seeds per (unit, task, rate): 100. Learning rates: 0.001, 0.01, 0.1.

## Success rate at each unit's best learning rate

| unit | T1_linear | T2_xor | T3_band | T4_stripes | T5_circles | T6_moons |
|---|---|---|---|---|---|---|
| perceptron | 100% (lr 0.01) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) |
| gaussian | 97% (lr 0.01) | 52% (lr 0.1) | 62% (lr 0.1) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) |
| quadratic | 100% (lr 0.01) | 100% (lr 0.01) | 100% (lr 0.01) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) |
| abs | 100% (lr 0.01) | 41% (lr 0.1) | 54% (lr 0.01) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) |
| cubic | 98% (lr 0.1) | 100% (lr 0.1) | 55% (lr 0.1) | 48% (lr 0.1) | 0% (lr 0.001) | 0% (lr 0.001) |
| sine | 100% (lr 0.01) | 100% (lr 0.01) | 31% (lr 0.01) | 98% (lr 0.01) | 0% (lr 0.001) | 0% (lr 0.001) |
| eml | 88% (lr 0.1) | 72% (lr 0.01) | 38% (lr 0.1) | 13% (lr 0.1) | 0% (lr 0.001) | 0% (lr 0.001) |

## Gap: EML minus best comparison unit (evaluated tasks)

| task | EML | best comparison | gap |
|---|---|---|---|
| T2_xor | 72% | quadratic 100% | -28 pts |
| T3_band | 38% | quadratic 100% | -62 pts |
| T4_stripes | 13% | sine 98% | -85 pts |

## Success rate at every learning rate

| unit | task | lr 0.001 | lr 0.01 | lr 0.1 |
|---|---|---|---|---|
| perceptron | T1_linear | 66% | 100% | 100% |
| perceptron | T2_xor | 0% | 0% | 0% |
| perceptron | T3_band | 0% | 0% | 0% |
| perceptron | T4_stripes | 0% | 0% | 0% |
| perceptron | T5_circles | 0% | 0% | 0% |
| perceptron | T6_moons | 0% | 0% | 0% |
| gaussian | T1_linear | 0% | 97% | 94% |
| gaussian | T2_xor | 51% | 51% | 52% |
| gaussian | T3_band | 2% | 60% | 62% |
| gaussian | T4_stripes | 0% | 0% | 0% |
| gaussian | T5_circles | 0% | 0% | 0% |
| gaussian | T6_moons | 0% | 0% | 0% |
| quadratic | T1_linear | 75% | 100% | 100% |
| quadratic | T2_xor | 92% | 100% | 100% |
| quadratic | T3_band | 56% | 100% | 99% |
| quadratic | T4_stripes | 0% | 0% | 0% |
| quadratic | T5_circles | 0% | 0% | 0% |
| quadratic | T6_moons | 0% | 0% | 0% |
| abs | T1_linear | 40% | 100% | 100% |
| abs | T2_xor | 39% | 38% | 41% |
| abs | T3_band | 50% | 54% | 54% |
| abs | T4_stripes | 0% | 0% | 0% |
| abs | T5_circles | 0% | 0% | 0% |
| abs | T6_moons | 0% | 0% | 0% |
| cubic | T1_linear | 53% | 96% | 98% |
| cubic | T2_xor | 82% | 99% | 100% |
| cubic | T3_band | 0% | 43% | 55% |
| cubic | T4_stripes | 24% | 35% | 48% |
| cubic | T5_circles | 0% | 0% | 0% |
| cubic | T6_moons | 0% | 0% | 0% |
| sine | T1_linear | 8% | 100% | 99% |
| sine | T2_xor | 97% | 100% | 100% |
| sine | T3_band | 26% | 31% | 26% |
| sine | T4_stripes | 66% | 98% | 96% |
| sine | T5_circles | 0% | 0% | 0% |
| sine | T6_moons | 0% | 0% | 0% |
| eml | T1_linear | 34% | 59% | 88% |
| eml | T2_xor | 40% | 72% | 67% |
| eml | T3_band | 0% | 1% | 38% |
| eml | T4_stripes | 0% | 12% | 13% |
| eml | T5_circles | 0% | 0% | 0% |
| eml | T6_moons | 0% | 0% | 0% |

## Median train / test accuracy

| unit | task | lr 0.001 | lr 0.01 | lr 0.1 |
|---|---|---|---|---|
| perceptron | T1_linear | 0.993 / 0.990 | 1.000 / 1.000 | 1.000 / 1.000 |
| perceptron | T2_xor | 0.500 / 0.500 | 0.500 / 0.500 | 0.500 / 0.500 |
| perceptron | T3_band | 0.670 / 0.650 | 0.670 / 0.650 | 0.670 / 0.650 |
| perceptron | T4_stripes | 0.555 / 0.545 | 0.555 / 0.545 | 0.555 / 0.545 |
| perceptron | T5_circles | 0.713 / 0.740 | 0.713 / 0.740 | 0.713 / 0.740 |
| perceptron | T6_moons | 0.877 / 0.875 | 0.887 / 0.890 | 0.887 / 0.890 |
| gaussian | T1_linear | 0.973 / 0.985 | 1.000 / 1.000 | 1.000 / 1.000 |
| gaussian | T2_xor | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| gaussian | T3_band | 0.973 / 0.973 | 1.000 / 1.000 | 1.000 / 1.000 |
| gaussian | T4_stripes | 0.882 / 0.824 | 0.887 / 0.825 | 0.887 / 0.825 |
| gaussian | T5_circles | 0.766 / 0.740 | 0.777 / 0.752 | 0.775 / 0.752 |
| gaussian | T6_moons | 0.877 / 0.875 | 0.890 / 0.890 | 0.890 / 0.890 |
| quadratic | T1_linear | 0.998 / 0.998 | 1.000 / 1.000 | 1.000 / 1.000 |
| quadratic | T2_xor | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| quadratic | T3_band | 0.995 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| quadratic | T4_stripes | 0.574 / 0.599 | 0.547 / 0.537 | 0.547 / 0.530 |
| quadratic | T5_circles | 0.775 / 0.755 | 0.775 / 0.755 | 0.775 / 0.755 |
| quadratic | T6_moons | 0.887 / 0.885 | 0.890 / 0.887 | 0.890 / 0.887 |
| abs | T1_linear | 0.985 / 0.990 | 1.000 / 1.000 | 1.000 / 1.000 |
| abs | T2_xor | 0.500 / 0.500 | 0.500 / 0.500 | 0.500 / 0.500 |
| abs | T3_band | 0.989 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| abs | T4_stripes | 0.578 / 0.605 | 0.578 / 0.605 | 0.645 / 0.610 |
| abs | T5_circles | 0.772 / 0.740 | 0.772 / 0.740 | 0.772 / 0.743 |
| abs | T6_moons | 0.882 / 0.877 | 0.887 / 0.890 | 0.887 / 0.890 |
| cubic | T1_linear | 0.998 / 0.996 | 1.000 / 1.000 | 1.000 / 1.000 |
| cubic | T2_xor | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| cubic | T3_band | 0.757 / 0.735 | 0.974 / 0.962 | 1.000 / 1.000 |
| cubic | T4_stripes | 0.615 / 0.618 | 0.627 / 0.615 | 0.644 / 0.616 |
| cubic | T5_circles | 0.755 / 0.740 | 0.777 / 0.755 | 0.784 / 0.776 |
| cubic | T6_moons | 0.882 / 0.877 | 0.887 / 0.890 | 0.890 / 0.887 |
| sine | T1_linear | 0.985 / 0.990 | 1.000 / 1.000 | 1.000 / 1.000 |
| sine | T2_xor | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| sine | T3_band | 0.765 / 0.718 | 0.757 / 0.720 | 0.757 / 0.718 |
| sine | T4_stripes | 0.995 / 0.990 | 1.000 / 1.000 | 1.000 / 1.000 |
| sine | T5_circles | 0.713 / 0.740 | 0.774 / 0.752 | 0.795 / 0.792 |
| sine | T6_moons | 0.885 / 0.879 | 0.887 / 0.890 | 0.887 / 0.890 |
| eml | T1_linear | 0.886 / 0.901 | 0.996 / 0.995 | 1.000 / 1.000 |
| eml | T2_xor | 0.979 / 0.974 | 1.000 / 1.000 | 1.000 / 1.000 |
| eml | T3_band | 0.670 / 0.650 | 0.670 / 0.650 | 0.670 / 0.650 |
| eml | T4_stripes | 0.569 / 0.581 | 0.587 / 0.623 | 0.568 / 0.576 |
| eml | T5_circles | 0.713 / 0.740 | 0.713 / 0.740 | 0.713 / 0.740 |
| eml | T6_moons | 0.866 / 0.849 | 0.882 / 0.882 | 0.890 / 0.887 |

## Stability (best-rate runs)

| unit | task | NaN share | share clipped >50% of steps | median clip_step_frac | EML median pole_guard_frac | EML exp_clamp_hits | EML min |s| | EML max |g'| |
|---|---|---|---|---|---|---|---|---|
| perceptron | T1_linear | 0% | 0% | 0.000 | – | – | – | – |
| perceptron | T2_xor | 0% | 0% | 0.000 | – | – | – | – |
| perceptron | T3_band | 0% | 0% | 0.000 | – | – | – | – |
| perceptron | T4_stripes | 0% | 0% | 0.000 | – | – | – | – |
| perceptron | T5_circles | 0% | 0% | 0.000 | – | – | – | – |
| perceptron | T6_moons | 0% | 0% | 0.000 | – | – | – | – |
| gaussian | T1_linear | 0% | 0% | 0.000 | – | – | – | – |
| gaussian | T2_xor | 0% | 0% | 0.000 | – | – | – | – |
| gaussian | T3_band | 0% | 0% | 0.000 | – | – | – | – |
| gaussian | T4_stripes | 0% | 0% | 0.000 | – | – | – | – |
| gaussian | T5_circles | 0% | 0% | 0.000 | – | – | – | – |
| gaussian | T6_moons | 0% | 0% | 0.000 | – | – | – | – |
| quadratic | T1_linear | 0% | 0% | 0.000 | – | – | – | – |
| quadratic | T2_xor | 0% | 0% | 0.000 | – | – | – | – |
| quadratic | T3_band | 0% | 0% | 0.000 | – | – | – | – |
| quadratic | T4_stripes | 0% | 0% | 0.000 | – | – | – | – |
| quadratic | T5_circles | 0% | 0% | 0.000 | – | – | – | – |
| quadratic | T6_moons | 0% | 0% | 0.000 | – | – | – | – |
| abs | T1_linear | 0% | 0% | 0.000 | – | – | – | – |
| abs | T2_xor | 0% | 0% | 0.000 | – | – | – | – |
| abs | T3_band | 0% | 0% | 0.000 | – | – | – | – |
| abs | T4_stripes | 0% | 0% | 0.000 | – | – | – | – |
| abs | T5_circles | 0% | 0% | 0.000 | – | – | – | – |
| abs | T6_moons | 0% | 0% | 0.000 | – | – | – | – |
| cubic | T1_linear | 0% | 0% | 0.000 | – | – | – | – |
| cubic | T2_xor | 0% | 0% | 0.000 | – | – | – | – |
| cubic | T3_band | 0% | 0% | 0.000 | – | – | – | – |
| cubic | T4_stripes | 0% | 0% | 0.000 | – | – | – | – |
| cubic | T5_circles | 0% | 0% | 0.000 | – | – | – | – |
| cubic | T6_moons | 0% | 0% | 0.000 | – | – | – | – |
| sine | T1_linear | 0% | 0% | 0.000 | – | – | – | – |
| sine | T2_xor | 0% | 0% | 0.000 | – | – | – | – |
| sine | T3_band | 0% | 0% | 0.000 | – | – | – | – |
| sine | T4_stripes | 0% | 0% | 0.000 | – | – | – | – |
| sine | T5_circles | 0% | 0% | 0.000 | – | – | – | – |
| sine | T6_moons | 0% | 0% | 0.000 | – | – | – | – |
| eml | T1_linear | 0% | 0% | 0.000 | 0 | 0 | 1.33e-06 | 2e+07 |
| eml | T2_xor | 0% | 0% | 0.000 | 0 | 0 | 1.89e-06 | 5.28e+05 |
| eml | T3_band | 0% | 0% | 0.000 | 0 | 0 | 1.2e-05 | 1.49e+25 |
| eml | T4_stripes | 0% | 0% | 0.000 | 0 | 0 | 5.04e-05 | 1.98e+04 |
| eml | T5_circles | 0% | 0% | 0.000 | 0 | 0 | 1.13e-05 | 8.86e+04 |
| eml | T6_moons | 0% | 0% | 0.000 | 0 | 0 | 4.35e-06 | 2.3e+05 |

## G1: boundary families

| unit | most lines | region pattern | width ratios seen |
|---|---|---|---|
| eml | 3 | A \| B \| A \| B | 3.67e-09 – 5.74 |
| cubic | 3 | A \| B \| A \| B | 0.0635 – 15.7 |
| sine | unlimited (13 in s∈(−20, 20)) | A \| B \| A \| B \| ... | 0.168 – 5.97 |
| perceptron | 1 | A \| B | n/a |
| gaussian | 2 | B \| A \| B | n/a |
| quadratic | 2 | B \| A \| B | n/a |
| abs | 2 | B \| A \| B | n/a |

**Limitation (t/k scale).** The threshold `t` and scale `k` work on different scales for different activations, so one Adam step moves some units' boundaries more than others'. Part of any reliability gap may come from parameter scale rather than from the activation's shape. Results at every learning rate are shown above so the effect is visible.

## Plots

- [plots/abs_T1_linear_best.png](plots/abs_T1_linear_best.png)
- [plots/abs_T1_linear_worst.png](plots/abs_T1_linear_worst.png)
- [plots/abs_T2_xor_best.png](plots/abs_T2_xor_best.png)
- [plots/abs_T2_xor_worst.png](plots/abs_T2_xor_worst.png)
- [plots/abs_T3_band_best.png](plots/abs_T3_band_best.png)
- [plots/abs_T3_band_worst.png](plots/abs_T3_band_worst.png)
- [plots/abs_T4_stripes_best.png](plots/abs_T4_stripes_best.png)
- [plots/abs_T4_stripes_worst.png](plots/abs_T4_stripes_worst.png)
- [plots/abs_T5_circles_best.png](plots/abs_T5_circles_best.png)
- [plots/abs_T5_circles_worst.png](plots/abs_T5_circles_worst.png)
- [plots/abs_T6_moons_best.png](plots/abs_T6_moons_best.png)
- [plots/abs_T6_moons_worst.png](plots/abs_T6_moons_worst.png)
- [plots/cubic_T1_linear_best.png](plots/cubic_T1_linear_best.png)
- [plots/cubic_T1_linear_worst.png](plots/cubic_T1_linear_worst.png)
- [plots/cubic_T2_xor_best.png](plots/cubic_T2_xor_best.png)
- [plots/cubic_T2_xor_worst.png](plots/cubic_T2_xor_worst.png)
- [plots/cubic_T3_band_best.png](plots/cubic_T3_band_best.png)
- [plots/cubic_T3_band_worst.png](plots/cubic_T3_band_worst.png)
- [plots/cubic_T4_stripes_best.png](plots/cubic_T4_stripes_best.png)
- [plots/cubic_T4_stripes_worst.png](plots/cubic_T4_stripes_worst.png)
- [plots/cubic_T5_circles_best.png](plots/cubic_T5_circles_best.png)
- [plots/cubic_T5_circles_worst.png](plots/cubic_T5_circles_worst.png)
- [plots/cubic_T6_moons_best.png](plots/cubic_T6_moons_best.png)
- [plots/cubic_T6_moons_worst.png](plots/cubic_T6_moons_worst.png)
- [plots/eml_T1_linear_best.png](plots/eml_T1_linear_best.png)
- [plots/eml_T1_linear_worst.png](plots/eml_T1_linear_worst.png)
- [plots/eml_T2_xor_best.png](plots/eml_T2_xor_best.png)
- [plots/eml_T2_xor_worst.png](plots/eml_T2_xor_worst.png)
- [plots/eml_T3_band_best.png](plots/eml_T3_band_best.png)
- [plots/eml_T3_band_worst.png](plots/eml_T3_band_worst.png)
- [plots/eml_T4_stripes_best.png](plots/eml_T4_stripes_best.png)
- [plots/eml_T4_stripes_worst.png](plots/eml_T4_stripes_worst.png)
- [plots/eml_T5_circles_best.png](plots/eml_T5_circles_best.png)
- [plots/eml_T5_circles_worst.png](plots/eml_T5_circles_worst.png)
- [plots/eml_T6_moons_best.png](plots/eml_T6_moons_best.png)
- [plots/eml_T6_moons_worst.png](plots/eml_T6_moons_worst.png)
- [plots/gaussian_T1_linear_best.png](plots/gaussian_T1_linear_best.png)
- [plots/gaussian_T1_linear_worst.png](plots/gaussian_T1_linear_worst.png)
- [plots/gaussian_T2_xor_best.png](plots/gaussian_T2_xor_best.png)
- [plots/gaussian_T2_xor_worst.png](plots/gaussian_T2_xor_worst.png)
- [plots/gaussian_T3_band_best.png](plots/gaussian_T3_band_best.png)
- [plots/gaussian_T3_band_worst.png](plots/gaussian_T3_band_worst.png)
- [plots/gaussian_T4_stripes_best.png](plots/gaussian_T4_stripes_best.png)
- [plots/gaussian_T4_stripes_worst.png](plots/gaussian_T4_stripes_worst.png)
- [plots/gaussian_T5_circles_best.png](plots/gaussian_T5_circles_best.png)
- [plots/gaussian_T5_circles_worst.png](plots/gaussian_T5_circles_worst.png)
- [plots/gaussian_T6_moons_best.png](plots/gaussian_T6_moons_best.png)
- [plots/gaussian_T6_moons_worst.png](plots/gaussian_T6_moons_worst.png)
- [plots/perceptron_T1_linear_best.png](plots/perceptron_T1_linear_best.png)
- [plots/perceptron_T1_linear_worst.png](plots/perceptron_T1_linear_worst.png)
- [plots/perceptron_T2_xor_best.png](plots/perceptron_T2_xor_best.png)
- [plots/perceptron_T2_xor_worst.png](plots/perceptron_T2_xor_worst.png)
- [plots/perceptron_T3_band_best.png](plots/perceptron_T3_band_best.png)
- [plots/perceptron_T3_band_worst.png](plots/perceptron_T3_band_worst.png)
- [plots/perceptron_T4_stripes_best.png](plots/perceptron_T4_stripes_best.png)
- [plots/perceptron_T4_stripes_worst.png](plots/perceptron_T4_stripes_worst.png)
- [plots/perceptron_T5_circles_best.png](plots/perceptron_T5_circles_best.png)
- [plots/perceptron_T5_circles_worst.png](plots/perceptron_T5_circles_worst.png)
- [plots/perceptron_T6_moons_best.png](plots/perceptron_T6_moons_best.png)
- [plots/perceptron_T6_moons_worst.png](plots/perceptron_T6_moons_worst.png)
- [plots/quadratic_T1_linear_best.png](plots/quadratic_T1_linear_best.png)
- [plots/quadratic_T1_linear_worst.png](plots/quadratic_T1_linear_worst.png)
- [plots/quadratic_T2_xor_best.png](plots/quadratic_T2_xor_best.png)
- [plots/quadratic_T2_xor_worst.png](plots/quadratic_T2_xor_worst.png)
- [plots/quadratic_T3_band_best.png](plots/quadratic_T3_band_best.png)
- [plots/quadratic_T3_band_worst.png](plots/quadratic_T3_band_worst.png)
- [plots/quadratic_T4_stripes_best.png](plots/quadratic_T4_stripes_best.png)
- [plots/quadratic_T4_stripes_worst.png](plots/quadratic_T4_stripes_worst.png)
- [plots/quadratic_T5_circles_best.png](plots/quadratic_T5_circles_best.png)
- [plots/quadratic_T5_circles_worst.png](plots/quadratic_T5_circles_worst.png)
- [plots/quadratic_T6_moons_best.png](plots/quadratic_T6_moons_best.png)
- [plots/quadratic_T6_moons_worst.png](plots/quadratic_T6_moons_worst.png)
- [plots/sine_T1_linear_best.png](plots/sine_T1_linear_best.png)
- [plots/sine_T1_linear_worst.png](plots/sine_T1_linear_worst.png)
- [plots/sine_T2_xor_best.png](plots/sine_T2_xor_best.png)
- [plots/sine_T2_xor_worst.png](plots/sine_T2_xor_worst.png)
- [plots/sine_T3_band_best.png](plots/sine_T3_band_best.png)
- [plots/sine_T3_band_worst.png](plots/sine_T3_band_worst.png)
- [plots/sine_T4_stripes_best.png](plots/sine_T4_stripes_best.png)
- [plots/sine_T4_stripes_worst.png](plots/sine_T4_stripes_worst.png)
- [plots/sine_T5_circles_best.png](plots/sine_T5_circles_best.png)
- [plots/sine_T5_circles_worst.png](plots/sine_T5_circles_worst.png)
- [plots/sine_T6_moons_best.png](plots/sine_T6_moons_best.png)
- [plots/sine_T6_moons_worst.png](plots/sine_T6_moons_worst.png)
- [plots/warn_T1_linear_0.1_54.png](plots/warn_T1_linear_0.1_54.png)
- [plots/warn_T1_linear_0.1_81.png](plots/warn_T1_linear_0.1_81.png)
- [plots/warn_T1_linear_0.1_91.png](plots/warn_T1_linear_0.1_91.png)
- [plots/success_rates.png](plots/success_rates.png)
