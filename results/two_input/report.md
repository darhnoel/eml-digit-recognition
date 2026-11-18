# EML two-input study: report

> **Pole warning** (does not change the verdict): best-rate EML successes with final max 1/|s2| > 1e+06:

> - U2_xor lr=0.1 seed=68: max 1/|s2| = 1.61e+06, min |s| = 6.22e-07 ([plot](plots/warn_U2_xor_0.1_68.png))

## Verdict: **NO-GO** (G5 rule 4)

EML did not lead the best comparison unit by 20 points on any evaluated task.

*This verdict concerns one real-valued, two-input EML unit, e^{s1} − ln|s2|, against seven two-input units of the same size. It does not decide multi-layer or complex-valued EML.*

Seeds per (unit, task, rate): 100. Learning rates: 0.001, 0.01, 0.1.

## Success rate at each unit's best learning rate

| unit | U1_linear | U2_xor | U3_disc | U4_parabola | U5_wave | U6_wedge | U7_moons | C1_ring | C2_checker |
|---|---|---|---|---|---|---|---|---|---|
| linear | 100% (lr 0.01) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) |
| parabola | 100% (lr 0.01) | 100% (lr 0.01) | 0% (lr 0.001) | 100% (lr 0.01) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) |
| ellipse | 100% (lr 0.01) | 100% (lr 0.01) | 100% (lr 0.1) | 100% (lr 0.01) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) |
| hyperbola | 100% (lr 0.01) | 100% (lr 0.01) | 0% (lr 0.001) | 100% (lr 0.01) | 0% (lr 0.001) | 100% (lr 0.1) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) |
| sine-wave | 100% (lr 0.01) | 100% (lr 0.01) | 0% (lr 0.001) | 87% (lr 0.1) | 99% (lr 0.1) | 98% (lr 0.1) | 98% (lr 0.1) | 0% (lr 0.001) | 0% (lr 0.001) |
| exp-half | 100% (lr 0.01) | 100% (lr 0.01) | 0% (lr 0.001) | 97% (lr 0.01) | 0% (lr 0.001) | 100% (lr 0.1) | 0% (lr 0.001) | 0% (lr 0.001) | 0% (lr 0.001) |
| log-half | 100% (lr 0.1) | 100% (lr 0.01) | 0% (lr 0.001) | 71% (lr 0.1) | 0% (lr 0.001) | 0% (lr 0.001) | 8% (lr 0.01) | 0% (lr 0.001) | 0% (lr 0.001) |
| eml | 100% (lr 0.01) | 99% (lr 0.1) | 0% (lr 0.001) | 66% (lr 0.1) | 0% (lr 0.001) | 65% (lr 0.1) | 6% (lr 0.01) | 0% (lr 0.001) | 0% (lr 0.001) |

## Gap: EML minus best comparison unit (evaluated tasks)

| task | EML | best comparison | gap |
|---|---|---|---|
| U2_xor | 99% | parabola 100% | -1 pts |
| U3_disc | 0% | ellipse 100% | -100 pts |
| U4_parabola | 66% | parabola 100% | -34 pts |
| U5_wave | 0% | sine-wave 99% | -99 pts |
| U6_wedge | 65% | hyperbola 100% | -35 pts |
| U7_moons | 6% | sine-wave 98% | -92 pts |

## Success rate at every learning rate

| unit | task | lr 0.001 | lr 0.01 | lr 0.1 |
|---|---|---|---|---|
| linear | U1_linear | 80% | 100% | 100% |
| linear | U2_xor | 0% | 0% | 0% |
| linear | U3_disc | 0% | 0% | 0% |
| linear | U4_parabola | 0% | 0% | 0% |
| linear | U5_wave | 0% | 0% | 0% |
| linear | U6_wedge | 0% | 0% | 0% |
| linear | U7_moons | 0% | 0% | 0% |
| linear | C1_ring | 0% | 0% | 0% |
| linear | C2_checker | 0% | 0% | 0% |
| parabola | U1_linear | 81% | 100% | 100% |
| parabola | U2_xor | 94% | 100% | 100% |
| parabola | U3_disc | 0% | 0% | 0% |
| parabola | U4_parabola | 55% | 100% | 97% |
| parabola | U5_wave | 0% | 0% | 0% |
| parabola | U6_wedge | 0% | 0% | 0% |
| parabola | U7_moons | 0% | 0% | 0% |
| parabola | C1_ring | 0% | 0% | 0% |
| parabola | C2_checker | 0% | 0% | 0% |
| ellipse | U1_linear | 88% | 100% | 100% |
| ellipse | U2_xor | 98% | 100% | 100% |
| ellipse | U3_disc | 66% | 97% | 100% |
| ellipse | U4_parabola | 60% | 100% | 100% |
| ellipse | U5_wave | 0% | 0% | 0% |
| ellipse | U6_wedge | 0% | 0% | 0% |
| ellipse | U7_moons | 0% | 0% | 0% |
| ellipse | C1_ring | 0% | 0% | 0% |
| ellipse | C2_checker | 0% | 0% | 0% |
| hyperbola | U1_linear | 92% | 100% | 100% |
| hyperbola | U2_xor | 99% | 100% | 100% |
| hyperbola | U3_disc | 0% | 0% | 0% |
| hyperbola | U4_parabola | 39% | 100% | 100% |
| hyperbola | U5_wave | 0% | 0% | 0% |
| hyperbola | U6_wedge | 59% | 99% | 100% |
| hyperbola | U7_moons | 0% | 0% | 0% |
| hyperbola | C1_ring | 0% | 0% | 0% |
| hyperbola | C2_checker | 0% | 0% | 0% |
| sine-wave | U1_linear | 84% | 100% | 100% |
| sine-wave | U2_xor | 99% | 100% | 100% |
| sine-wave | U3_disc | 0% | 0% | 0% |
| sine-wave | U4_parabola | 0% | 63% | 87% |
| sine-wave | U5_wave | 5% | 98% | 99% |
| sine-wave | U6_wedge | 0% | 52% | 98% |
| sine-wave | U7_moons | 2% | 75% | 98% |
| sine-wave | C1_ring | 0% | 0% | 0% |
| sine-wave | C2_checker | 0% | 0% | 0% |
| exp-half | U1_linear | 77% | 100% | 100% |
| exp-half | U2_xor | 73% | 100% | 100% |
| exp-half | U3_disc | 0% | 0% | 0% |
| exp-half | U4_parabola | 43% | 97% | 96% |
| exp-half | U5_wave | 0% | 0% | 0% |
| exp-half | U6_wedge | 3% | 99% | 100% |
| exp-half | U7_moons | 0% | 0% | 0% |
| exp-half | C1_ring | 0% | 0% | 0% |
| exp-half | C2_checker | 0% | 0% | 0% |
| log-half | U1_linear | 37% | 82% | 100% |
| log-half | U2_xor | 34% | 100% | 100% |
| log-half | U3_disc | 0% | 0% | 0% |
| log-half | U4_parabola | 0% | 0% | 71% |
| log-half | U5_wave | 0% | 0% | 0% |
| log-half | U6_wedge | 0% | 0% | 0% |
| log-half | U7_moons | 2% | 8% | 4% |
| log-half | C1_ring | 0% | 0% | 0% |
| log-half | C2_checker | 0% | 0% | 0% |
| eml | U1_linear | 45% | 100% | 100% |
| eml | U2_xor | 63% | 93% | 99% |
| eml | U3_disc | 0% | 0% | 0% |
| eml | U4_parabola | 0% | 49% | 66% |
| eml | U5_wave | 0% | 0% | 0% |
| eml | U6_wedge | 0% | 39% | 65% |
| eml | U7_moons | 1% | 6% | 2% |
| eml | C1_ring | 0% | 0% | 0% |
| eml | C2_checker | 0% | 0% | 0% |

## Median train / test accuracy

| unit | task | lr 0.001 | lr 0.01 | lr 0.1 |
|---|---|---|---|---|
| linear | U1_linear | 0.995 / 0.993 | 1.000 / 1.000 | 1.000 / 1.000 |
| linear | U2_xor | 0.500 / 0.500 | 0.500 / 0.500 | 0.500 / 0.500 |
| linear | U3_disc | 0.713 / 0.740 | 0.713 / 0.740 | 0.713 / 0.740 |
| linear | U4_parabola | 0.802 / 0.848 | 0.802 / 0.848 | 0.802 / 0.848 |
| linear | U5_wave | 0.943 / 0.955 | 0.945 / 0.958 | 0.945 / 0.958 |
| linear | U6_wedge | 0.818 / 0.752 | 0.818 / 0.752 | 0.818 / 0.752 |
| linear | U7_moons | 0.887 / 0.882 | 0.887 / 0.890 | 0.887 / 0.890 |
| linear | C1_ring | 0.708 / 0.710 | 0.708 / 0.710 | 0.708 / 0.710 |
| linear | C2_checker | 0.502 / 0.460 | 0.502 / 0.460 | 0.502 / 0.460 |
| parabola | U1_linear | 0.998 / 0.998 | 1.000 / 1.000 | 1.000 / 1.000 |
| parabola | U2_xor | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| parabola | U3_disc | 0.795 / 0.757 | 0.795 / 0.757 | 0.795 / 0.757 |
| parabola | U4_parabola | 0.990 / 0.995 | 1.000 / 1.000 | 1.000 / 1.000 |
| parabola | U5_wave | 0.938 / 0.958 | 0.943 / 0.958 | 0.945 / 0.958 |
| parabola | U6_wedge | 0.965 / 0.965 | 0.973 / 0.975 | 0.973 / 0.975 |
| parabola | U7_moons | 0.890 / 0.887 | 0.890 / 0.890 | 0.887 / 0.887 |
| parabola | C1_ring | 0.672 / 0.660 | 0.670 / 0.657 | 0.670 / 0.657 |
| parabola | C2_checker | 0.504 / 0.475 | 0.502 / 0.460 | 0.502 / 0.460 |
| ellipse | U1_linear | 0.998 / 0.998 | 1.000 / 1.000 | 1.000 / 1.000 |
| ellipse | U2_xor | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| ellipse | U3_disc | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| ellipse | U4_parabola | 0.990 / 0.983 | 1.000 / 0.996 | 1.000 / 0.998 |
| ellipse | U5_wave | 0.938 / 0.955 | 0.940 / 0.958 | 0.943 / 0.958 |
| ellipse | U6_wedge | 0.958 / 0.932 | 0.968 / 0.968 | 0.970 / 0.971 |
| ellipse | U7_moons | 0.887 / 0.885 | 0.890 / 0.887 | 0.890 / 0.887 |
| ellipse | C1_ring | 0.711 / 0.701 | 0.795 / 0.823 | 0.795 / 0.823 |
| ellipse | C2_checker | 0.530 / 0.512 | 0.502 / 0.468 | 0.677 / 0.645 |
| hyperbola | U1_linear | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| hyperbola | U2_xor | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| hyperbola | U3_disc | 0.780 / 0.755 | 0.787 / 0.757 | 0.790 / 0.755 |
| hyperbola | U4_parabola | 0.985 / 0.993 | 1.000 / 1.000 | 1.000 / 1.000 |
| hyperbola | U5_wave | 0.943 / 0.955 | 0.950 / 0.958 | 0.950 / 0.958 |
| hyperbola | U6_wedge | 0.993 / 0.993 | 1.000 / 1.000 | 1.000 / 1.000 |
| hyperbola | U7_moons | 0.887 / 0.890 | 0.890 / 0.890 | 0.890 / 0.890 |
| hyperbola | C1_ring | 0.680 / 0.685 | 0.670 / 0.682 | 0.685 / 0.685 |
| hyperbola | C2_checker | 0.632 / 0.635 | 0.654 / 0.644 | 0.657 / 0.645 |
| sine-wave | U1_linear | 0.995 / 0.995 | 1.000 / 1.000 | 1.000 / 1.000 |
| sine-wave | U2_xor | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| sine-wave | U3_disc | 0.713 / 0.740 | 0.782 / 0.752 | 0.713 / 0.740 |
| sine-wave | U4_parabola | 0.964 / 0.983 | 0.990 / 0.995 | 1.000 / 1.000 |
| sine-wave | U5_wave | 0.943 / 0.955 | 1.000 / 1.000 | 1.000 / 1.000 |
| sine-wave | U6_wedge | 0.828 / 0.755 | 0.995 / 0.998 | 1.000 / 0.998 |
| sine-wave | U7_moons | 0.887 / 0.887 | 1.000 / 1.000 | 1.000 / 1.000 |
| sine-wave | C1_ring | 0.708 / 0.710 | 0.693 / 0.690 | 0.708 / 0.710 |
| sine-wave | C2_checker | 0.635 / 0.643 | 0.682 / 0.662 | 0.682 / 0.662 |
| exp-half | U1_linear | 1.000 / 0.998 | 1.000 / 1.000 | 1.000 / 1.000 |
| exp-half | U2_xor | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| exp-half | U3_disc | 0.777 / 0.752 | 0.787 / 0.760 | 0.790 / 0.760 |
| exp-half | U4_parabola | 0.975 / 0.983 | 1.000 / 1.000 | 1.000 / 1.000 |
| exp-half | U5_wave | 0.938 / 0.955 | 0.940 / 0.958 | 0.945 / 0.960 |
| exp-half | U6_wedge | 0.953 / 0.943 | 1.000 / 0.995 | 1.000 / 0.998 |
| exp-half | U7_moons | 0.887 / 0.885 | 0.890 / 0.887 | 0.882 / 0.882 |
| exp-half | C1_ring | 0.693 / 0.680 | 0.680 / 0.688 | 0.685 / 0.688 |
| exp-half | C2_checker | 0.505 / 0.485 | 0.502 / 0.460 | 0.502 / 0.460 |
| log-half | U1_linear | 0.930 / 0.933 | 1.000 / 0.999 | 1.000 / 1.000 |
| log-half | U2_xor | 0.968 / 0.965 | 1.000 / 0.998 | 1.000 / 1.000 |
| log-half | U3_disc | 0.713 / 0.740 | 0.785 / 0.756 | 0.787 / 0.777 |
| log-half | U4_parabola | 0.814 / 0.848 | 0.930 / 0.946 | 1.000 / 1.000 |
| log-half | U5_wave | 0.935 / 0.948 | 0.940 / 0.955 | 0.938 / 0.955 |
| log-half | U6_wedge | 0.779 / 0.715 | 0.839 / 0.771 | 0.978 / 0.973 |
| log-half | U7_moons | 0.882 / 0.881 | 0.890 / 0.885 | 0.890 / 0.887 |
| log-half | C1_ring | 0.708 / 0.710 | 0.703 / 0.680 | 0.703 / 0.680 |
| log-half | C2_checker | 0.547 / 0.532 | 0.570 / 0.547 | 0.530 / 0.486 |
| eml | U1_linear | 0.983 / 0.978 | 1.000 / 1.000 | 1.000 / 1.000 |
| eml | U2_xor | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| eml | U3_disc | 0.713 / 0.740 | 0.792 / 0.760 | 0.795 / 0.760 |
| eml | U4_parabola | 0.826 / 0.850 | 0.954 / 0.971 | 0.996 / 0.999 |
| eml | U5_wave | 0.935 / 0.950 | 0.938 / 0.955 | 0.938 / 0.955 |
| eml | U6_wedge | 0.819 / 0.767 | 0.868 / 0.833 | 1.000 / 0.998 |
| eml | U7_moons | 0.885 / 0.880 | 0.895 / 0.885 | 0.895 / 0.885 |
| eml | C1_ring | 0.688 / 0.689 | 0.699 / 0.680 | 0.705 / 0.688 |
| eml | C2_checker | 0.554 / 0.540 | 0.590 / 0.570 | 0.599 / 0.577 |

## Stability (best-rate runs)

| unit | task | NaN share | share clipped >50% of steps | median clip_step_frac | EML median pole_guard_frac | EML exp_clamp_hits | EML min |s2| | EML max 1/|s2| |
|---|---|---|---|---|---|---|---|---|
| linear | U1_linear | 0% | 0% | 0.000 | – | – | – | – |
| linear | U2_xor | 0% | 0% | 0.000 | – | – | – | – |
| linear | U3_disc | 0% | 0% | 0.000 | – | – | – | – |
| linear | U4_parabola | 0% | 0% | 0.000 | – | – | – | – |
| linear | U5_wave | 0% | 0% | 0.000 | – | – | – | – |
| linear | U6_wedge | 0% | 0% | 0.000 | – | – | – | – |
| linear | U7_moons | 0% | 0% | 0.000 | – | – | – | – |
| linear | C1_ring | 0% | 0% | 0.000 | – | – | – | – |
| linear | C2_checker | 0% | 0% | 0.000 | – | – | – | – |
| parabola | U1_linear | 0% | 0% | 0.000 | – | – | – | – |
| parabola | U2_xor | 0% | 0% | 0.000 | – | – | – | – |
| parabola | U3_disc | 0% | 0% | 0.000 | – | – | – | – |
| parabola | U4_parabola | 0% | 0% | 0.000 | – | – | – | – |
| parabola | U5_wave | 0% | 0% | 0.000 | – | – | – | – |
| parabola | U6_wedge | 0% | 0% | 0.000 | – | – | – | – |
| parabola | U7_moons | 0% | 0% | 0.000 | – | – | – | – |
| parabola | C1_ring | 0% | 0% | 0.000 | – | – | – | – |
| parabola | C2_checker | 0% | 0% | 0.000 | – | – | – | – |
| ellipse | U1_linear | 0% | 0% | 0.000 | – | – | – | – |
| ellipse | U2_xor | 0% | 0% | 0.000 | – | – | – | – |
| ellipse | U3_disc | 0% | 0% | 0.000 | – | – | – | – |
| ellipse | U4_parabola | 0% | 0% | 0.000 | – | – | – | – |
| ellipse | U5_wave | 0% | 0% | 0.000 | – | – | – | – |
| ellipse | U6_wedge | 0% | 0% | 0.000 | – | – | – | – |
| ellipse | U7_moons | 0% | 0% | 0.000 | – | – | – | – |
| ellipse | C1_ring | 0% | 0% | 0.000 | – | – | – | – |
| ellipse | C2_checker | 0% | 0% | 0.000 | – | – | – | – |
| hyperbola | U1_linear | 0% | 0% | 0.000 | – | – | – | – |
| hyperbola | U2_xor | 0% | 0% | 0.000 | – | – | – | – |
| hyperbola | U3_disc | 0% | 0% | 0.000 | – | – | – | – |
| hyperbola | U4_parabola | 0% | 0% | 0.000 | – | – | – | – |
| hyperbola | U5_wave | 0% | 0% | 0.000 | – | – | – | – |
| hyperbola | U6_wedge | 0% | 0% | 0.000 | – | – | – | – |
| hyperbola | U7_moons | 0% | 0% | 0.000 | – | – | – | – |
| hyperbola | C1_ring | 0% | 0% | 0.000 | – | – | – | – |
| hyperbola | C2_checker | 0% | 0% | 0.000 | – | – | – | – |
| sine-wave | U1_linear | 0% | 0% | 0.000 | – | – | – | – |
| sine-wave | U2_xor | 0% | 0% | 0.000 | – | – | – | – |
| sine-wave | U3_disc | 0% | 0% | 0.000 | – | – | – | – |
| sine-wave | U4_parabola | 0% | 0% | 0.000 | – | – | – | – |
| sine-wave | U5_wave | 0% | 0% | 0.000 | – | – | – | – |
| sine-wave | U6_wedge | 0% | 0% | 0.000 | – | – | – | – |
| sine-wave | U7_moons | 0% | 0% | 0.000 | – | – | – | – |
| sine-wave | C1_ring | 0% | 0% | 0.000 | – | – | – | – |
| sine-wave | C2_checker | 0% | 0% | 0.000 | – | – | – | – |
| exp-half | U1_linear | 0% | 0% | 0.000 | – | – | – | – |
| exp-half | U2_xor | 0% | 0% | 0.000 | – | – | – | – |
| exp-half | U3_disc | 0% | 0% | 0.000 | – | – | – | – |
| exp-half | U4_parabola | 0% | 0% | 0.000 | – | – | – | – |
| exp-half | U5_wave | 0% | 0% | 0.000 | – | – | – | – |
| exp-half | U6_wedge | 0% | 0% | 0.000 | – | – | – | – |
| exp-half | U7_moons | 0% | 0% | 0.000 | – | – | – | – |
| exp-half | C1_ring | 0% | 0% | 0.000 | – | – | – | – |
| exp-half | C2_checker | 0% | 0% | 0.000 | – | – | – | – |
| log-half | U1_linear | 0% | 0% | 0.000 | – | – | – | – |
| log-half | U2_xor | 0% | 0% | 0.000 | – | – | – | – |
| log-half | U3_disc | 0% | 0% | 0.000 | – | – | – | – |
| log-half | U4_parabola | 0% | 0% | 0.000 | – | – | – | – |
| log-half | U5_wave | 0% | 0% | 0.000 | – | – | – | – |
| log-half | U6_wedge | 0% | 0% | 0.000 | – | – | – | – |
| log-half | U7_moons | 0% | 0% | 0.000 | – | – | – | – |
| log-half | C1_ring | 0% | 0% | 0.000 | – | – | – | – |
| log-half | C2_checker | 0% | 0% | 0.000 | – | – | – | – |
| eml | U1_linear | 0% | 0% | 0.000 | 0 | 0 | 3.61e-05 | 2.77e+04 |
| eml | U2_xor | 0% | 0% | 0.000 | 0 | 0 | 6.22e-07 | 1.61e+06 |
| eml | U3_disc | 0% | 0% | 0.000 | 0 | 0 | 1.08e-05 | 9.25e+04 |
| eml | U4_parabola | 0% | 0% | 0.000 | 0 | 0 | 0.00025 | 4.01e+03 |
| eml | U5_wave | 0% | 0% | 0.000 | 0 | 0 | 2e-06 | 5.01e+05 |
| eml | U6_wedge | 0% | 0% | 0.000 | 0 | 0 | 3.31e-05 | 3.02e+04 |
| eml | U7_moons | 0% | 0% | 0.000 | 0 | 0 | 1.71e-06 | 5.84e+05 |
| eml | C1_ring | 0% | 0% | 0.000 | 0 | 0 | 4.4e-06 | 2.27e+05 |
| eml | C2_checker | 0% | 0% | 0.000 | 0 | 0 | 2.12e-05 | 4.71e+04 |

## G1: boundary families

| unit | g_a(s1) | g_b(s2) | boundary in (s1, s2) |
|---|---|---|---|
| linear | s1 | s2 | straight line |
| parabola | s1² | s2 | parabola |
| ellipse | s1² | −s2² | ellipse (closed) |
| hyperbola | s1² | s2² | hyperbola; two crossing lines at t = 0 |
| sine-wave | sin s1 | s2 | wavy curve |
| exp-half | e^s1 | s2 | exponential curve |
| log-half | s1 | ln\|s2\| | funnel, half-width e^(s1−t) |
| eml | e^s1 | ln\|s2\| | funnel, half-width exp(e^s1 − t) |

**Limitation (t/k scale).** The threshold `t` and scale `k` work on different scales for different activations, so one Adam step moves some units' boundaries more than others'. Part of any reliability gap may come from parameter scale rather than from the activation's shape. Results at every learning rate are shown above so the effect is visible.

## Plots

- [plots/ellipse_C1_ring_best.png](plots/ellipse_C1_ring_best.png)
- [plots/ellipse_C1_ring_worst.png](plots/ellipse_C1_ring_worst.png)
- [plots/ellipse_C2_checker_best.png](plots/ellipse_C2_checker_best.png)
- [plots/ellipse_C2_checker_worst.png](plots/ellipse_C2_checker_worst.png)
- [plots/ellipse_U1_linear_best.png](plots/ellipse_U1_linear_best.png)
- [plots/ellipse_U1_linear_worst.png](plots/ellipse_U1_linear_worst.png)
- [plots/ellipse_U2_xor_best.png](plots/ellipse_U2_xor_best.png)
- [plots/ellipse_U2_xor_worst.png](plots/ellipse_U2_xor_worst.png)
- [plots/ellipse_U3_disc_best.png](plots/ellipse_U3_disc_best.png)
- [plots/ellipse_U3_disc_worst.png](plots/ellipse_U3_disc_worst.png)
- [plots/ellipse_U4_parabola_best.png](plots/ellipse_U4_parabola_best.png)
- [plots/ellipse_U4_parabola_worst.png](plots/ellipse_U4_parabola_worst.png)
- [plots/ellipse_U5_wave_best.png](plots/ellipse_U5_wave_best.png)
- [plots/ellipse_U5_wave_worst.png](plots/ellipse_U5_wave_worst.png)
- [plots/ellipse_U6_wedge_best.png](plots/ellipse_U6_wedge_best.png)
- [plots/ellipse_U6_wedge_worst.png](plots/ellipse_U6_wedge_worst.png)
- [plots/ellipse_U7_moons_best.png](plots/ellipse_U7_moons_best.png)
- [plots/ellipse_U7_moons_worst.png](plots/ellipse_U7_moons_worst.png)
- [plots/eml_C1_ring_best.png](plots/eml_C1_ring_best.png)
- [plots/eml_C1_ring_worst.png](plots/eml_C1_ring_worst.png)
- [plots/eml_C2_checker_best.png](plots/eml_C2_checker_best.png)
- [plots/eml_C2_checker_worst.png](plots/eml_C2_checker_worst.png)
- [plots/eml_U1_linear_best.png](plots/eml_U1_linear_best.png)
- [plots/eml_U1_linear_worst.png](plots/eml_U1_linear_worst.png)
- [plots/eml_U2_xor_best.png](plots/eml_U2_xor_best.png)
- [plots/eml_U2_xor_worst.png](plots/eml_U2_xor_worst.png)
- [plots/eml_U3_disc_best.png](plots/eml_U3_disc_best.png)
- [plots/eml_U3_disc_worst.png](plots/eml_U3_disc_worst.png)
- [plots/eml_U4_parabola_best.png](plots/eml_U4_parabola_best.png)
- [plots/eml_U4_parabola_worst.png](plots/eml_U4_parabola_worst.png)
- [plots/eml_U5_wave_best.png](plots/eml_U5_wave_best.png)
- [plots/eml_U5_wave_worst.png](plots/eml_U5_wave_worst.png)
- [plots/eml_U6_wedge_best.png](plots/eml_U6_wedge_best.png)
- [plots/eml_U6_wedge_worst.png](plots/eml_U6_wedge_worst.png)
- [plots/eml_U7_moons_best.png](plots/eml_U7_moons_best.png)
- [plots/eml_U7_moons_worst.png](plots/eml_U7_moons_worst.png)
- [plots/exp-half_C1_ring_best.png](plots/exp-half_C1_ring_best.png)
- [plots/exp-half_C1_ring_worst.png](plots/exp-half_C1_ring_worst.png)
- [plots/exp-half_C2_checker_best.png](plots/exp-half_C2_checker_best.png)
- [plots/exp-half_C2_checker_worst.png](plots/exp-half_C2_checker_worst.png)
- [plots/exp-half_U1_linear_best.png](plots/exp-half_U1_linear_best.png)
- [plots/exp-half_U1_linear_worst.png](plots/exp-half_U1_linear_worst.png)
- [plots/exp-half_U2_xor_best.png](plots/exp-half_U2_xor_best.png)
- [plots/exp-half_U2_xor_worst.png](plots/exp-half_U2_xor_worst.png)
- [plots/exp-half_U3_disc_best.png](plots/exp-half_U3_disc_best.png)
- [plots/exp-half_U3_disc_worst.png](plots/exp-half_U3_disc_worst.png)
- [plots/exp-half_U4_parabola_best.png](plots/exp-half_U4_parabola_best.png)
- [plots/exp-half_U4_parabola_worst.png](plots/exp-half_U4_parabola_worst.png)
- [plots/exp-half_U5_wave_best.png](plots/exp-half_U5_wave_best.png)
- [plots/exp-half_U5_wave_worst.png](plots/exp-half_U5_wave_worst.png)
- [plots/exp-half_U6_wedge_best.png](plots/exp-half_U6_wedge_best.png)
- [plots/exp-half_U6_wedge_worst.png](plots/exp-half_U6_wedge_worst.png)
- [plots/exp-half_U7_moons_best.png](plots/exp-half_U7_moons_best.png)
- [plots/exp-half_U7_moons_worst.png](plots/exp-half_U7_moons_worst.png)
- [plots/hyperbola_C1_ring_best.png](plots/hyperbola_C1_ring_best.png)
- [plots/hyperbola_C1_ring_worst.png](plots/hyperbola_C1_ring_worst.png)
- [plots/hyperbola_C2_checker_best.png](plots/hyperbola_C2_checker_best.png)
- [plots/hyperbola_C2_checker_worst.png](plots/hyperbola_C2_checker_worst.png)
- [plots/hyperbola_U1_linear_best.png](plots/hyperbola_U1_linear_best.png)
- [plots/hyperbola_U1_linear_worst.png](plots/hyperbola_U1_linear_worst.png)
- [plots/hyperbola_U2_xor_best.png](plots/hyperbola_U2_xor_best.png)
- [plots/hyperbola_U2_xor_worst.png](plots/hyperbola_U2_xor_worst.png)
- [plots/hyperbola_U3_disc_best.png](plots/hyperbola_U3_disc_best.png)
- [plots/hyperbola_U3_disc_worst.png](plots/hyperbola_U3_disc_worst.png)
- [plots/hyperbola_U4_parabola_best.png](plots/hyperbola_U4_parabola_best.png)
- [plots/hyperbola_U4_parabola_worst.png](plots/hyperbola_U4_parabola_worst.png)
- [plots/hyperbola_U5_wave_best.png](plots/hyperbola_U5_wave_best.png)
- [plots/hyperbola_U5_wave_worst.png](plots/hyperbola_U5_wave_worst.png)
- [plots/hyperbola_U6_wedge_best.png](plots/hyperbola_U6_wedge_best.png)
- [plots/hyperbola_U6_wedge_worst.png](plots/hyperbola_U6_wedge_worst.png)
- [plots/hyperbola_U7_moons_best.png](plots/hyperbola_U7_moons_best.png)
- [plots/hyperbola_U7_moons_worst.png](plots/hyperbola_U7_moons_worst.png)
- [plots/linear_C1_ring_best.png](plots/linear_C1_ring_best.png)
- [plots/linear_C1_ring_worst.png](plots/linear_C1_ring_worst.png)
- [plots/linear_C2_checker_best.png](plots/linear_C2_checker_best.png)
- [plots/linear_C2_checker_worst.png](plots/linear_C2_checker_worst.png)
- [plots/linear_U1_linear_best.png](plots/linear_U1_linear_best.png)
- [plots/linear_U1_linear_worst.png](plots/linear_U1_linear_worst.png)
- [plots/linear_U2_xor_best.png](plots/linear_U2_xor_best.png)
- [plots/linear_U2_xor_worst.png](plots/linear_U2_xor_worst.png)
- [plots/linear_U3_disc_best.png](plots/linear_U3_disc_best.png)
- [plots/linear_U3_disc_worst.png](plots/linear_U3_disc_worst.png)
- [plots/linear_U4_parabola_best.png](plots/linear_U4_parabola_best.png)
- [plots/linear_U4_parabola_worst.png](plots/linear_U4_parabola_worst.png)
- [plots/linear_U5_wave_best.png](plots/linear_U5_wave_best.png)
- [plots/linear_U5_wave_worst.png](plots/linear_U5_wave_worst.png)
- [plots/linear_U6_wedge_best.png](plots/linear_U6_wedge_best.png)
- [plots/linear_U6_wedge_worst.png](plots/linear_U6_wedge_worst.png)
- [plots/linear_U7_moons_best.png](plots/linear_U7_moons_best.png)
- [plots/linear_U7_moons_worst.png](plots/linear_U7_moons_worst.png)
- [plots/log-half_C1_ring_best.png](plots/log-half_C1_ring_best.png)
- [plots/log-half_C1_ring_worst.png](plots/log-half_C1_ring_worst.png)
- [plots/log-half_C2_checker_best.png](plots/log-half_C2_checker_best.png)
- [plots/log-half_C2_checker_worst.png](plots/log-half_C2_checker_worst.png)
- [plots/log-half_U1_linear_best.png](plots/log-half_U1_linear_best.png)
- [plots/log-half_U1_linear_worst.png](plots/log-half_U1_linear_worst.png)
- [plots/log-half_U2_xor_best.png](plots/log-half_U2_xor_best.png)
- [plots/log-half_U2_xor_worst.png](plots/log-half_U2_xor_worst.png)
- [plots/log-half_U3_disc_best.png](plots/log-half_U3_disc_best.png)
- [plots/log-half_U3_disc_worst.png](plots/log-half_U3_disc_worst.png)
- [plots/log-half_U4_parabola_best.png](plots/log-half_U4_parabola_best.png)
- [plots/log-half_U4_parabola_worst.png](plots/log-half_U4_parabola_worst.png)
- [plots/log-half_U5_wave_best.png](plots/log-half_U5_wave_best.png)
- [plots/log-half_U5_wave_worst.png](plots/log-half_U5_wave_worst.png)
- [plots/log-half_U6_wedge_best.png](plots/log-half_U6_wedge_best.png)
- [plots/log-half_U6_wedge_worst.png](plots/log-half_U6_wedge_worst.png)
- [plots/log-half_U7_moons_best.png](plots/log-half_U7_moons_best.png)
- [plots/log-half_U7_moons_worst.png](plots/log-half_U7_moons_worst.png)
- [plots/parabola_C1_ring_best.png](plots/parabola_C1_ring_best.png)
- [plots/parabola_C1_ring_worst.png](plots/parabola_C1_ring_worst.png)
- [plots/parabola_C2_checker_best.png](plots/parabola_C2_checker_best.png)
- [plots/parabola_C2_checker_worst.png](plots/parabola_C2_checker_worst.png)
- [plots/parabola_U1_linear_best.png](plots/parabola_U1_linear_best.png)
- [plots/parabola_U1_linear_worst.png](plots/parabola_U1_linear_worst.png)
- [plots/parabola_U2_xor_best.png](plots/parabola_U2_xor_best.png)
- [plots/parabola_U2_xor_worst.png](plots/parabola_U2_xor_worst.png)
- [plots/parabola_U3_disc_best.png](plots/parabola_U3_disc_best.png)
- [plots/parabola_U3_disc_worst.png](plots/parabola_U3_disc_worst.png)
- [plots/parabola_U4_parabola_best.png](plots/parabola_U4_parabola_best.png)
- [plots/parabola_U4_parabola_worst.png](plots/parabola_U4_parabola_worst.png)
- [plots/parabola_U5_wave_best.png](plots/parabola_U5_wave_best.png)
- [plots/parabola_U5_wave_worst.png](plots/parabola_U5_wave_worst.png)
- [plots/parabola_U6_wedge_best.png](plots/parabola_U6_wedge_best.png)
- [plots/parabola_U6_wedge_worst.png](plots/parabola_U6_wedge_worst.png)
- [plots/parabola_U7_moons_best.png](plots/parabola_U7_moons_best.png)
- [plots/parabola_U7_moons_worst.png](plots/parabola_U7_moons_worst.png)
- [plots/sine-wave_C1_ring_best.png](plots/sine-wave_C1_ring_best.png)
- [plots/sine-wave_C1_ring_worst.png](plots/sine-wave_C1_ring_worst.png)
- [plots/sine-wave_C2_checker_best.png](plots/sine-wave_C2_checker_best.png)
- [plots/sine-wave_C2_checker_worst.png](plots/sine-wave_C2_checker_worst.png)
- [plots/sine-wave_U1_linear_best.png](plots/sine-wave_U1_linear_best.png)
- [plots/sine-wave_U1_linear_worst.png](plots/sine-wave_U1_linear_worst.png)
- [plots/sine-wave_U2_xor_best.png](plots/sine-wave_U2_xor_best.png)
- [plots/sine-wave_U2_xor_worst.png](plots/sine-wave_U2_xor_worst.png)
- [plots/sine-wave_U3_disc_best.png](plots/sine-wave_U3_disc_best.png)
- [plots/sine-wave_U3_disc_worst.png](plots/sine-wave_U3_disc_worst.png)
- [plots/sine-wave_U4_parabola_best.png](plots/sine-wave_U4_parabola_best.png)
- [plots/sine-wave_U4_parabola_worst.png](plots/sine-wave_U4_parabola_worst.png)
- [plots/sine-wave_U5_wave_best.png](plots/sine-wave_U5_wave_best.png)
- [plots/sine-wave_U5_wave_worst.png](plots/sine-wave_U5_wave_worst.png)
- [plots/sine-wave_U6_wedge_best.png](plots/sine-wave_U6_wedge_best.png)
- [plots/sine-wave_U6_wedge_worst.png](plots/sine-wave_U6_wedge_worst.png)
- [plots/sine-wave_U7_moons_best.png](plots/sine-wave_U7_moons_best.png)
- [plots/sine-wave_U7_moons_worst.png](plots/sine-wave_U7_moons_worst.png)
- [plots/warn_U2_xor_0.1_68.png](plots/warn_U2_xor_0.1_68.png)
- [plots/success_rates.png](plots/success_rates.png)
