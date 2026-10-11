# Plan revision 3: pre-registered tests on independent farms

Gain = 100 (1 - RMSE_A / RMSE_B), backbone mean over LSTM, GRU, TCN, Transformer, DLinear, PatchTST, iTransformer; RMSE from seed-averaged squared errors. CI = joint 7-day block bootstrap (2000 reps). p = registered two-sided group t-test: the statistic is recomputed within each of q = 8 contiguous groups of the test windows (H2: groups with >= 50 flat-high window-steps). Calibration of the test: reports/r3_placebo_sdwpf.md.

## H1: backbone + PAF vs backbone (G > 0)

| farm | H | windows | G % [95% CI] | groups G>0 | t | p | p Holm (family of 6) |
|---|---|---|---|---|---|---|---|
| penmanshiel | 4 | 12843 | -0.75 [-1.12, -0.38] | 0/8 | -5.41 | 0.0010 | 0.0060 |
| penmanshiel | 8 | 12819 | -0.18 [-0.55, +0.19] | 3/8 | -1.28 | 0.2401 | 0.2401 |
| penmanshiel | 16 | 12771 | -0.46 [-0.83, -0.11] | 1/8 | -3.01 | 0.0198 | 0.0714 |
| kelmarsh | 4 | 12888 | -1.12 [-1.73, -0.56] | 0/8 | -4.97 | 0.0016 | 0.0081 |
| kelmarsh | 8 | 12852 | -1.17 [-2.08, -0.46] | 0/8 | -2.64 | 0.0332 | 0.0714 |
| kelmarsh | 16 | 12780 | -1.11 [-1.85, -0.53] | 1/8 | -3.08 | 0.0178 | 0.0714 |

## H2: regime complementarity, theta = gain(PAF vs ens2 | flat-high) - gain(PAF vs ens2 | steep) > 0

| farm | H | flat-high % | gain flat-high % | gain steep % | theta [95% CI] | testable | groups used | groups theta>0 | t | p | p Holm (family of 6) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| penmanshiel | 4 | 9.0 | +2.44 | -0.60 | +3.04 [+1.66, +4.30] | yes | 7 | 7 | 2.87 | 0.0285 | 0.1427 |
| penmanshiel | 8 | 9.0 | +0.59 | -1.51 | +2.10 [+0.94, +3.28] | yes | 8 | 8 | 2.24 | 0.0598 | 0.2390 |
| penmanshiel | 16 | 9.0 | +1.24 | -2.76 | +4.00 [+3.26, +4.78] | yes | 8 | 8 | 10.04 | 0.0000 | 0.0001 |
| kelmarsh | 4 | 5.3 | +5.06 | -1.44 | +6.50 [+3.55, +9.18] | NO | 5 | - | - | - (not testable) | 1.0000 |
| kelmarsh | 8 | 5.3 | +6.24 | -2.03 | +8.27 [+6.21, +10.69] | NO | 5 | - | - | - (not testable) | 1.0000 |
| kelmarsh | 16 | 5.4 | +4.64 | -2.40 | +7.04 [+4.83, +10.47] | NO | 5 | - | - | - (not testable) | 1.0000 |

## Descriptive: backbone-mean gains by region (%)

### penmanshiel, H = 4

| region | share % | PAF vs ens2 | PAF vs backbone | ens2 vs backbone |
|---|---|---|---|---|
| ex post flat-low | 33.1 | +7.03 [+6.13, +7.94] | +2.65 [+1.58, +3.67] | -5.74 [-6.96, -4.67] |
| ex post steep | 57.9 | -0.60 [-0.90, -0.30] | -1.10 [-1.41, -0.78] | -0.49 [-0.81, -0.17] |
| ex post flat-high | 9.0 | +2.44 [+1.04, +3.68] | -0.51 [-2.12, +0.94] | -3.09 [-4.28, -1.89] |
| ex ante flat-low | 33.1 | +2.10 [+1.36, +3.01] | -0.84 [-1.43, -0.25] | -3.35 [-4.46, -2.45] |
| ex ante steep | 57.9 | -0.07 [-0.39, +0.25] | -1.01 [-1.35, -0.69] | -0.93 [-1.27, -0.58] |
| ex ante flat-high | 9.0 | +3.63 [+2.35, +4.52] | +1.45 [+0.12, +2.52] | -2.58 [-4.04, -1.02] |
| all | 100.0 | +0.51 [+0.12, +0.94] | -0.75 [-1.12, -0.38] | -1.28 [-1.65, -0.91] |

| backbone | seeds | RMSE bb | RMSE +PAF | RMSE +ens2 | PAF vs bb % | PAF vs ens2 % | ens2 vs bb % |
|---|---|---|---|---|---|---|---|
| LSTM | 3 | 2.989 | 2.956 | 2.963 | +1.09 | +0.23 | +0.86 |
| GRU | 3 | 2.985 | 2.963 | 2.974 | +0.74 | +0.38 | +0.36 |
| TCN | 3 | 3.002 | 3.009 | 3.012 | -0.24 | +0.09 | -0.33 |
| Transformer | 3 | 2.999 | 2.980 | 2.986 | +0.64 | +0.22 | +0.42 |
| DLinear | 3 | 3.102 | 3.153 | 3.185 | -1.64 | +1.00 | -2.67 |
| PatchTST | 3 | 3.080 | 3.246 | 3.304 | -5.39 | +1.74 | -7.25 |
| iTransformer | 3 | 3.116 | 3.130 | 3.128 | -0.45 | -0.07 | -0.38 |
| TimeMixer (separate) | 3 | 3.103 | 3.129 | 3.121 | -0.85 | -0.28 | -0.56 |

Proposed (val-selected) GRU+paf vs strongest baseline (val-selected) GRU: RMSE 2.963 vs 2.985, gain +0.74% [+0.17, +1.31], p (group t) 0.0130 (descriptive).
Diverged runs (val RMSE > 3.937): none

### penmanshiel, H = 8

| region | share % | PAF vs ens2 | PAF vs backbone | ens2 vs backbone |
|---|---|---|---|---|
| ex post flat-low | 33.1 | +4.41 [+3.42, +5.50] | +3.96 [+3.14, +4.76] | -0.60 [-1.60, +0.17] |
| ex post steep | 57.9 | -1.51 [-1.76, -1.25] | -0.52 [-0.85, -0.19] | +0.98 [+0.64, +1.30] |
| ex post flat-high | 9.0 | +0.59 [-0.48, +1.70] | -1.09 [-2.37, +0.18] | -1.75 [-3.15, -0.66] |
| ex ante flat-low | 33.2 | -0.48 [-1.20, +0.16] | -0.44 [-0.96, +0.12] | +0.06 [-0.51, +0.69] |
| ex ante steep | 57.9 | -0.85 [-1.10, -0.56] | -0.31 [-0.67, +0.04] | +0.54 [+0.18, +0.87] |
| ex ante flat-high | 8.9 | +1.45 [+0.27, +2.54] | +1.02 [-0.29, +1.87] | -0.46 [-1.66, +0.57] |
| all | 100.0 | -0.56 [-0.81, -0.25] | -0.18 [-0.55, +0.19] | +0.39 [+0.03, +0.72] |

| backbone | seeds | RMSE bb | RMSE +PAF | RMSE +ens2 | PAF vs bb % | PAF vs ens2 % | ens2 vs bb % |
|---|---|---|---|---|---|---|---|
| LSTM | 3 | 3.805 | 3.748 | 3.750 | +1.50 | +0.06 | +1.44 |
| GRU | 3 | 3.802 | 3.752 | 3.764 | +1.31 | +0.33 | +0.99 |
| TCN | 3 | 3.772 | 3.777 | 3.771 | -0.13 | -0.15 | +0.02 |
| Transformer | 3 | 3.845 | 3.770 | 3.765 | +1.95 | -0.13 | +2.08 |
| DLinear | 3 | 3.888 | 3.912 | 3.928 | -0.64 | +0.39 | -1.03 |
| PatchTST | 3 | 3.889 | 4.052 | 3.931 | -4.18 | -3.07 | -1.08 |
| iTransformer | 3 | 3.936 | 3.977 | 3.924 | -1.05 | -1.35 | +0.29 |
| TimeMixer (separate) | 3 | 3.908 | 3.942 | 3.887 | -0.88 | -1.41 | +0.53 |

Proposed (val-selected) Transformer+paf vs strongest baseline (val-selected) Transformer: RMSE 3.770 vs 3.845, gain +1.95% [+0.84, +3.07], p (group t) 0.0399 (descriptive).
Diverged runs (val RMSE > 4.820): none

### penmanshiel, H = 16

| region | share % | PAF vs ens2 | PAF vs backbone | ens2 vs backbone |
|---|---|---|---|---|
| ex post flat-low | 33.1 | +1.96 [+0.63, +3.49] | +1.47 [+0.36, +2.64] | -0.49 [-1.50, +0.47] |
| ex post steep | 57.9 | -2.76 [-3.09, -2.47] | -0.89 [-1.34, -0.45] | +1.81 [+1.41, +2.23] |
| ex post flat-high | 9.0 | +1.24 [+0.57, +1.94] | -0.22 [-1.05, +0.55] | -1.53 [-2.43, -0.77] |
| ex ante flat-low | 33.2 | -0.94 [-1.52, -0.21] | -0.55 [-1.13, +0.10] | +0.39 [-0.11, +0.90] |
| ex ante steep | 57.9 | -1.50 [-1.86, -1.12] | -0.59 [-0.99, -0.22] | +0.89 [+0.49, +1.34] |
| ex ante flat-high | 8.9 | -0.78 [-1.94, +0.32] | +0.44 [-0.98, +1.69] | +1.22 [+0.33, +2.05] |
| all | 100.0 | -1.35 [-1.69, -0.99] | -0.46 [-0.83, -0.11] | +0.88 [+0.48, +1.29] |

| backbone | seeds | RMSE bb | RMSE +PAF | RMSE +ens2 | PAF vs bb % | PAF vs ens2 % | ens2 vs bb % |
|---|---|---|---|---|---|---|---|
| LSTM | 3 | 4.756 | 4.835 | 4.748 | -1.67 | -1.84 | +0.17 |
| GRU | 3 | 4.818 | 4.745 | 4.732 | +1.51 | -0.28 | +1.78 |
| TCN | 3 | 4.722 | 4.705 | 4.695 | +0.35 | -0.22 | +0.58 |
| Transformer | 3 | 4.890 | 4.847 | 4.843 | +0.89 | -0.07 | +0.97 |
| DLinear | 3 | 4.842 | 4.847 | 4.818 | -0.10 | -0.58 | +0.48 |
| PatchTST | 3 | 4.891 | 5.026 | 4.850 | -2.76 | -3.62 | +0.83 |
| iTransformer | 3 | 4.949 | 5.021 | 4.883 | -1.47 | -2.83 | +1.32 |
| TimeMixer (separate) | 3 | 4.916 | 4.966 | 4.816 | -1.01 | -3.11 | +2.03 |

Proposed (val-selected) Transformer+paf vs strongest baseline (val-selected) TCN: RMSE 4.847 vs 4.722, gain -2.64% [-4.45, -0.83], p (group t) 0.0861 (descriptive).
Diverged runs (val RMSE > 5.979): none

### kelmarsh, H = 4

| region | share % | PAF vs ens2 | PAF vs backbone | ens2 vs backbone |
|---|---|---|---|---|
| ex post flat-low | 30.7 | +1.68 [+0.21, +2.96] | -0.73 [-3.14, +0.71] | -2.47 [-5.48, -0.43] |
| ex post steep | 64.0 | -1.44 [-1.78, -1.06] | -1.58 [-2.10, -1.11] | -0.14 [-0.47, +0.15] |
| ex post flat-high | 5.3 | +5.06 [+2.19, +7.73] | +4.43 [+0.69, +6.36] | -0.76 [-2.72, +0.41] |
| ex ante flat-low | 30.7 | +0.81 [+0.12, +1.63] | -0.44 [-3.56, +1.33] | -1.27 [-4.99, +0.64] |
| ex ante steep | 63.9 | -0.70 [-1.04, -0.33] | -0.99 [-1.39, -0.63] | -0.29 [-0.51, -0.08] |
| ex ante flat-high | 5.3 | -4.57 [-14.85, +0.90] | -4.55 [-17.05, +2.17] | -0.01 [-3.26, +2.19] |
| all | 100.0 | -0.79 [-1.17, -0.40] | -1.12 [-1.73, -0.56] | -0.34 [-0.76, +0.02] |

| backbone | seeds | RMSE bb | RMSE +PAF | RMSE +ens2 | PAF vs bb % | PAF vs ens2 % | ens2 vs bb % |
|---|---|---|---|---|---|---|---|
| LSTM | 3 | 1.076 | 1.093 | 1.084 | -1.60 | -0.83 | -0.77 |
| GRU | 3 | 1.067 | 1.087 | 1.070 | -1.80 | -1.54 | -0.25 |
| TCN | 3 | 1.078 | 1.077 | 1.080 | +0.14 | +0.34 | -0.20 |
| Transformer | 3 | 1.077 | 1.081 | 1.071 | -0.41 | -0.95 | +0.53 |
| DLinear | 3 | 1.077 | 1.093 | 1.092 | -1.46 | -0.02 | -1.43 |
| PatchTST | 3 | 1.080 | 1.091 | 1.082 | -1.00 | -0.86 | -0.14 |
| iTransformer | 3 | 1.084 | 1.103 | 1.085 | -1.74 | -1.65 | -0.08 |
| TimeMixer (separate) | 3 | 1.079 | 1.096 | 1.096 | -1.58 | -0.00 | -1.57 |

Proposed (val-selected) Transformer+paf vs strongest baseline (val-selected) GRU: RMSE 1.081 vs 1.067, gain -1.28% [-3.01, -0.12], p (group t) 0.0659 (descriptive).
Diverged runs (val RMSE > 1.514): none

### kelmarsh, H = 8

| region | share % | PAF vs ens2 | PAF vs backbone | ens2 vs backbone |
|---|---|---|---|---|
| ex post flat-low | 30.6 | +1.80 [+0.35, +2.98] | -1.67 [-4.64, +0.15] | -3.63 [-6.90, -1.38] |
| ex post steep | 64.0 | -2.03 [-2.48, -1.58] | -1.66 [-2.46, -0.98] | +0.35 [-0.28, +0.87] |
| ex post flat-high | 5.3 | +6.24 [+4.27, +8.62] | +4.64 [+3.38, +6.33] | -1.75 [-4.04, +0.19] |
| ex ante flat-low | 30.7 | -0.17 [-0.89, +0.69] | -1.39 [-5.67, +0.94] | -1.22 [-5.20, +1.00] |
| ex ante steep | 64.0 | -0.90 [-1.23, -0.53] | -0.96 [-1.57, -0.49] | -0.06 [-0.62, +0.33] |
| ex ante flat-high | 5.3 | -3.74 [-12.29, +1.55] | -4.20 [-14.17, +2.26] | -0.49 [-3.25, +1.59] |
| all | 100.0 | -1.01 [-1.48, -0.54] | -1.17 [-2.08, -0.46] | -0.16 [-0.94, +0.36] |

| backbone | seeds | RMSE bb | RMSE +PAF | RMSE +ens2 | PAF vs bb % | PAF vs ens2 % | ens2 vs bb % |
|---|---|---|---|---|---|---|---|
| LSTM | 3 | 1.309 | 1.330 | 1.321 | -1.57 | -0.69 | -0.87 |
| GRU | 3 | 1.301 | 1.323 | 1.304 | -1.71 | -1.43 | -0.28 |
| TCN | 3 | 1.308 | 1.309 | 1.311 | -0.02 | +0.17 | -0.18 |
| Transformer | 3 | 1.336 | 1.352 | 1.327 | -1.15 | -1.88 | +0.72 |
| DLinear | 3 | 1.310 | 1.324 | 1.321 | -1.07 | -0.22 | -0.85 |
| PatchTST | 3 | 1.319 | 1.336 | 1.320 | -1.32 | -1.27 | -0.05 |
| iTransformer | 3 | 1.330 | 1.348 | 1.325 | -1.32 | -1.71 | +0.38 |
| TimeMixer (separate) | 3 | 1.317 | 1.337 | 1.310 | -1.53 | -2.03 | +0.49 |

Proposed (val-selected) GRU+paf vs strongest baseline (val-selected) GRU: RMSE 1.323 vs 1.301, gain -1.71% [-2.91, -0.78], p (group t) 0.0111 (descriptive).
Diverged runs (val RMSE > 1.826): none

### kelmarsh, H = 16

| region | share % | PAF vs ens2 | PAF vs backbone | ens2 vs backbone |
|---|---|---|---|---|
| ex post flat-low | 30.5 | +0.21 [-1.99, +2.58] | -0.87 [-2.17, +0.24] | -1.12 [-4.35, +1.46] |
| ex post steep | 64.2 | -2.40 [-3.07, -1.70] | -1.87 [-2.73, -1.13] | +0.50 [-0.18, +1.04] |
| ex post flat-high | 5.4 | +4.64 [+2.75, +7.93] | +3.92 [+2.73, +6.14] | -0.78 [-2.81, +0.79] |
| ex ante flat-low | 30.7 | +0.38 [-0.97, +2.26] | -0.05 [-1.69, +1.24] | -0.46 [-3.92, +1.70] |
| ex ante steep | 63.9 | -1.30 [-1.73, -0.78] | -0.92 [-1.43, -0.46] | +0.37 [-0.17, +0.81] |
| ex ante flat-high | 5.4 | -4.17 [-10.30, -0.18] | -4.70 [-12.89, +0.72] | -0.53 [-3.18, +1.84] |
| all | 100.0 | -1.34 [-1.85, -0.77] | -1.11 [-1.85, -0.53] | +0.23 [-0.60, +0.84] |

| backbone | seeds | RMSE bb | RMSE +PAF | RMSE +ens2 | PAF vs bb % | PAF vs ens2 % | ens2 vs bb % |
|---|---|---|---|---|---|---|---|
| LSTM | 3 | 1.607 | 1.625 | 1.619 | -1.11 | -0.35 | -0.75 |
| GRU | 3 | 1.596 | 1.626 | 1.603 | -1.89 | -1.48 | -0.40 |
| TCN | 3 | 1.609 | 1.612 | 1.606 | -0.18 | -0.37 | +0.19 |
| Transformer | 3 | 1.649 | 1.655 | 1.637 | -0.41 | -1.13 | +0.72 |
| DLinear | 3 | 1.606 | 1.621 | 1.612 | -0.92 | -0.56 | -0.36 |
| PatchTST | 3 | 1.629 | 1.656 | 1.612 | -1.63 | -2.68 | +1.02 |
| iTransformer | 3 | 1.647 | 1.673 | 1.627 | -1.62 | -2.82 | +1.17 |
| TimeMixer (separate) | 3 | 1.625 | 1.650 | 1.605 | -1.59 | -2.82 | +1.20 |

Proposed (val-selected) GRU+paf vs strongest baseline (val-selected) GRU: RMSE 1.626 vs 1.596, gain -1.89% [-3.50, -0.62], p (group t) 0.0392 (descriptive).
Diverged runs (val RMSE > 2.240): none

